import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

try:
    from sklearn.ensemble import RandomForestRegressor
except ModuleNotFoundError:
    RandomForestRegressor = None

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:
    XGBRegressor = None


st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
)


DEFAULT_TICKERS = ["SPY", "GOLD", "TLT", "IEFA", "USO", "VNQ"]


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min())


def performance_metrics(returns: pd.Series) -> pd.Series:
    annual_return = (1 + returns).prod() ** (12 / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else np.nan

    return pd.Series(
        {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown(returns),
        }
    )


@st.cache_data(show_spinner=False)
def download_prices(tickers: list[str], start_date, end_date) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    return prices.dropna(how="all").ffill().dropna(axis=1)


def equal_weight_returns(monthly_returns: pd.DataFrame) -> pd.Series:
    weights = pd.Series(1 / len(monthly_returns.columns), index=monthly_returns.columns)
    return monthly_returns @ weights


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float,
) -> pd.Series:
    tickers = list(expected_returns.index)
    n_assets = len(tickers)
    mu = expected_returns.to_numpy()
    cov = covariance.loc[tickers, tickers].to_numpy()

    def negative_sharpe(weights):
        portfolio_return = weights @ mu
        portfolio_risk = np.sqrt(weights @ cov @ weights)
        if portfolio_risk == 0:
            return 0
        return -portfolio_return / portfolio_risk

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, max_weight) for _ in range(n_assets))
    initial = np.repeat(1 / n_assets, n_assets)

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return pd.Series(initial, index=tickers)

    return pd.Series(result.x, index=tickers)


def make_features(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker]
        frame = pd.DataFrame(index=returns.index)
        frame["Ticker"] = ticker
        frame["Return_1M"] = r
        frame["Momentum_3M"] = r.rolling(3).sum()
        frame["Momentum_6M"] = r.rolling(6).sum()
        frame["Momentum_12M"] = r.rolling(12).sum()
        frame["Volatility_3M"] = r.rolling(3).std()
        frame["Volatility_6M"] = r.rolling(6).std()
        frame["Target_Next_Return"] = r.shift(-1)
        rows.append(frame)

    return pd.concat(rows).dropna()


def normalize_long_only(predictions: pd.Series, max_weight: float) -> pd.Series:
    positive = predictions.clip(lower=0)
    if positive.sum() <= 0:
        return pd.Series(1 / len(predictions), index=predictions.index)

    weights = positive / positive.sum()
    weights = weights.clip(upper=max_weight)
    return weights / weights.sum()


def machine_learning_returns(
    monthly_returns: pd.DataFrame,
    model_name: str,
    lookback_months: int,
    max_weight: float,
) -> pd.Series:
    features = make_features(monthly_returns)
    feature_columns = [
        "Return_1M",
        "Momentum_3M",
        "Momentum_6M",
        "Momentum_12M",
        "Volatility_3M",
        "Volatility_6M",
    ]

    records = []
    for current_date in monthly_returns.index[lookback_months:-1]:
        next_date = monthly_returns.index[monthly_returns.index.get_loc(current_date) + 1]
        train_data = features[features.index < current_date]
        predict_data = features[features.index == current_date]

        if train_data.empty or predict_data.empty:
            continue

        if model_name == "Random Forest":
            if RandomForestRegressor is None:
                continue
            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        else:
            if XGBRegressor is None:
                continue
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
            )

        model.fit(train_data[feature_columns], train_data["Target_Next_Return"])

        predictions = pd.Series(
            model.predict(predict_data[feature_columns]),
            index=predict_data["Ticker"],
        ).reindex(monthly_returns.columns).dropna()

        weights = normalize_long_only(predictions, max_weight=max_weight)
        realized = monthly_returns.loc[next_date, weights.index]
        records.append({"Date": next_date, model_name: weights @ realized})

    if not records:
        return pd.Series(dtype=float, name=model_name)

    return pd.DataFrame(records).set_index("Date")[model_name]


st.title("Dynamic Portfolio Optimization Dashboard")
st.caption("Equal Weight, Mean-Variance, Random Forest, and XGBoost comparison")

with st.sidebar:
    st.header("Settings")
    ticker_text = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS))
    tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
    start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp.today())
    lookback_months = st.slider("Training lookback months", 24, 60, 36, 1)
    max_weight = st.slider("Maximum weight per asset", 0.10, 0.60, 0.35, 0.05)
    run = st.button("Run Dashboard", type="primary")


if not run:
    st.info("Choose the settings from the sidebar, then click Run Dashboard.")
    st.stop()

with st.spinner("Downloading data and building portfolios..."):
    prices = download_prices(tickers, start_date, end_date)
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    equal_returns = equal_weight_returns(monthly_returns)
    mv_weights = mean_variance_weights(
        monthly_returns.mean(),
        monthly_returns.cov(),
        max_weight=max_weight,
    )
    mv_returns = monthly_returns @ mv_weights

    comparison = pd.DataFrame(
        {
            "Equal Weight": equal_returns,
            "Mean-Variance": mv_returns,
        }
    )

    rf_returns = machine_learning_returns(
        monthly_returns,
        model_name="Random Forest",
        lookback_months=lookback_months,
        max_weight=max_weight,
    )
    if not rf_returns.empty:
        comparison["Random Forest"] = rf_returns

    xgb_returns = machine_learning_returns(
        monthly_returns,
        model_name="XGBoost",
        lookback_months=lookback_months,
        max_weight=max_weight,
    )
    if not xgb_returns.empty:
        comparison["XGBoost"] = xgb_returns

    comparison = comparison.dropna()
    metrics = comparison.apply(performance_metrics).T
    cumulative = (1 + comparison).cumprod()


st.subheader("Adjusted Closing Prices")
st.line_chart(prices)

st.subheader("Portfolio Performance")
st.line_chart(cumulative)

st.subheader("Performance Metrics")
st.dataframe(metrics.style.format("{:.4f}"), use_container_width=True)

st.subheader("Mean-Variance Weights")
st.dataframe(mv_weights.to_frame("Weight").style.format("{:.2%}"), use_container_width=True)

missing_models = []
if RandomForestRegressor is None:
    missing_models.append("scikit-learn")
if XGBRegressor is None:
    missing_models.append("xgboost")

if missing_models:
    st.warning(
        "Some machine-learning models were skipped because these packages are missing: "
        + ", ".join(missing_models)
        + ". Install them with: pip install scikit-learn xgboost"
    )
