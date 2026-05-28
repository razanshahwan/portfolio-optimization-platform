import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ModuleNotFoundError:
    sm = None
    variance_inflation_factor = None

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    RandomForestClassifier = None
    RandomForestRegressor = None
    LogisticRegression = None
    TimeSeriesSplit = None
    StandardScaler = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ModuleNotFoundError:
    XGBClassifier = None
    XGBRegressor = None


st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")

DEFAULT_TICKERS = ["XOM", "GOLD", "JNJ", "AAPL"]
DEFAULT_TICKER_TEXT = ", ".join(DEFAULT_TICKERS)
OLD_DEFAULT_TICKER_TEXT = "SPY, GOLD, TLT, IEFA, USO, VNQ"
SECTOR_TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
FEATURE_COLUMNS = ["Return_1M", "Momentum_3M", "Momentum_6M", "Momentum_12M", "Volatility_3M", "Volatility_6M"]


@st.cache_data(show_spinner=False)
def download_market_data(tickers, start_date, end_date):
    tickers = list(dict.fromkeys(tickers))
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
        volume = data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else pd.DataFrame()
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        volume = data[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in data else pd.DataFrame()

    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.reindex(columns=tickers).dropna(how="all").ffill().dropna(axis=1)
    if not volume.empty:
        volume = volume.loc[:, ~volume.columns.duplicated()]
        volume = volume.reindex(columns=prices.columns).dropna(how="all")
    return prices, volume


@st.cache_data(show_spinner=False)
def download_fundamentals(tickers):
    fields = [
        "shortName",
        "quoteType",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "dividendYield",
        "beta",
        "profitMargins",
        "returnOnEquity",
        "totalRevenue",
        "debtToEquity",
    ]
    rows = []
    for ticker in tickers:
        row = {"Ticker": ticker}
        try:
            info = yf.Ticker(ticker).info
            row.update({field: info.get(field, np.nan) for field in fields})
        except Exception as exc:
            row.update({field: np.nan for field in fields})
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(show_spinner=False)
def download_asset_intelligence(tickers):
    rows = []
    news_rows = []
    earnings_rows = []
    dividend_rows = []

    for ticker in tickers:
        asset = yf.Ticker(ticker)
        try:
            info = asset.info
        except Exception:
            info = {}
        try:
            fast = dict(asset.fast_info)
        except Exception:
            fast = {}

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or fast.get("last_price")
            or fast.get("lastPrice")
        )
        market_cap = info.get("marketCap") or fast.get("market_cap")

        rows.append(
            {
                "Ticker": ticker,
                "Name": info.get("shortName") or info.get("longName") or ticker,
                "Quote Type": info.get("quoteType") or info.get("typeDisp"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Website": info.get("website"),
                "Beta": info.get("beta"),
                "Dividend Yield": info.get("dividendYield"),
                "Dividend Rate": info.get("dividendRate"),
                "Payout Ratio": info.get("payoutRatio"),
                "Ex-Dividend Date": info.get("exDividendDate"),
                "Trailing EPS": info.get("trailingEps"),
                "Forward EPS": info.get("forwardEps"),
                "Price To Book": info.get("priceToBook"),
                "Recommendation": info.get("recommendationKey"),
                "Mean Analyst Rating": info.get("recommendationMean"),
                "Target Mean Price": info.get("targetMeanPrice"),
                "Current Price": current_price,
                "Market Cap": market_cap,
            }
        )

        try:
            dividends = asset.dividends.tail(8)
            for date, value in dividends.items():
                dividend_rows.append({"Ticker": ticker, "Date": date, "Dividend": value})
        except Exception:
            pass

        try:
            calendar = asset.calendar
            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                flat = calendar.T.reset_index()
                flat.columns = ["Metric", "Value"]
                for _, row in flat.iterrows():
                    earnings_rows.append({"Ticker": ticker, "Metric": row["Metric"], "Value": row["Value"]})
            elif isinstance(calendar, dict):
                for key, value in calendar.items():
                    earnings_rows.append({"Ticker": ticker, "Metric": key, "Value": value})
        except Exception:
            pass

        try:
            news = asset.news[:5]
            for item in news:
                content = item.get("content", item)
                title = content.get("title") or item.get("title")
                link = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link")
                provider = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
                pub_date = content.get("pubDate") or item.get("providerPublishTime")
                news_rows.append({"Ticker": ticker, "Title": title, "Provider": provider, "Published": pub_date, "URL": link})
        except Exception:
            pass

    return (
        pd.DataFrame(rows).set_index("Ticker"),
        pd.DataFrame(dividend_rows),
        pd.DataFrame(earnings_rows),
        pd.DataFrame(news_rows),
    )


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    return (cumulative / cumulative.cummax() - 1).min()


def performance_metrics(returns, periods_per_year=12):
    returns = returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    annual_return = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    return pd.Series(
        {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown(returns),
        }
    )


def equal_weight_weights(asset_names):
    return pd.Series(1 / len(asset_names), index=asset_names)


def normalize_long_only(scores, max_weight):
    positive = scores.clip(lower=0)
    if positive.sum() <= 0:
        return equal_weight_weights(list(scores.index))
    weights = positive / positive.sum()
    weights = weights.clip(upper=max_weight)
    return weights / weights.sum()


def mean_variance_weights(expected_returns, covariance, max_weight):
    asset_names = list(expected_returns.index)
    n_assets = len(asset_names)
    mu = expected_returns.to_numpy()
    cov = covariance.loc[asset_names, asset_names].to_numpy()

    def negative_sharpe(weights):
        portfolio_return = weights @ mu
        portfolio_risk = np.sqrt(weights @ cov @ weights)
        return -portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    result = minimize(
        negative_sharpe,
        np.repeat(1 / n_assets, n_assets),
        method="SLSQP",
        bounds=tuple((0, max_weight) for _ in range(n_assets)),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
    )
    if not result.success:
        return equal_weight_weights(asset_names)
    return pd.Series(result.x, index=asset_names)


def make_features(returns):
    returns = returns.loc[:, ~returns.columns.duplicated()].copy()
    rows = []
    for ticker in returns.columns:
        r = returns[ticker]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[:, 0]
        frame = pd.DataFrame(index=returns.index)
        frame["Ticker"] = ticker
        frame["Return_1M"] = r
        frame["Momentum_3M"] = r.rolling(3).sum()
        frame["Momentum_6M"] = r.rolling(6).sum()
        frame["Momentum_12M"] = r.rolling(12).sum()
        frame["Volatility_3M"] = r.rolling(3).std()
        frame["Volatility_6M"] = r.rolling(6).std()
        frame["Target_Next_Return"] = r.shift(-1)
        frame["Target_Up"] = (frame["Target_Next_Return"] > 0).astype(int)
        rows.append(frame)
    return pd.concat(rows).dropna()


def add_technical_indicators(price_series, short_window=20, long_window=50, rsi_period=14):
    frame = pd.DataFrame({"Close": price_series.dropna()})
    frame[f"SMA_{short_window}"] = frame["Close"].rolling(short_window).mean()
    frame[f"SMA_{long_window}"] = frame["Close"].rolling(long_window).mean()
    frame[f"EMA_{short_window}"] = frame["Close"].ewm(span=short_window, adjust=False).mean()
    frame[f"EMA_{long_window}"] = frame["Close"].ewm(span=long_window, adjust=False).mean()

    delta = frame["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss
    frame[f"RSI_{rsi_period}"] = 100 - (100 / (1 + rs))

    ema_12 = frame["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = frame["Close"].ewm(span=26, adjust=False).mean()
    frame["MACD"] = ema_12 - ema_26
    frame["MACD_Signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()

    rolling_mean = frame["Close"].rolling(20).mean()
    rolling_std = frame["Close"].rolling(20).std()
    frame["Bollinger_Upper"] = rolling_mean + 2 * rolling_std
    frame["Bollinger_Lower"] = rolling_mean - 2 * rolling_std
    return frame


def forecast_next_returns(monthly_returns, model_name, max_weight, n_estimators=100):
    features = make_features(monthly_returns)
    latest_date = monthly_returns.index.max()
    predict = features[features.index == latest_date]
    train = features[features.index < latest_date]

    if train.empty or predict.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if model_name == "Random Forest":
        if RandomForestRegressor is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5, random_state=42, n_jobs=-1)
    else:
        if XGBRegressor is None:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    model.fit(train[FEATURE_COLUMNS], train["Target_Next_Return"])
    predictions = pd.Series(model.predict(predict[FEATURE_COLUMNS]), index=predict["Ticker"])
    predictions = predictions.groupby(level=0).mean().reindex(monthly_returns.columns).dropna()
    weights = normalize_long_only(predictions, max_weight=max_weight)
    return predictions, weights


def detect_outliers(returns):
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        for date, value in outliers.items():
            rows.append({"Date": date, "Ticker": ticker, "Return": value, "Lower Bound": lower, "Upper Bound": upper})
    return pd.DataFrame(rows)


def run_portfolio_backtest(monthly_returns, lookback_months, max_weight, include_ml=True, n_estimators=100):
    features = make_features(monthly_returns)
    models = {}
    if include_ml and RandomForestRegressor is not None:
        models["Random Forest"] = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if include_ml and XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    records = []
    weights_records = []
    for current_date in monthly_returns.index[lookback_months:-1]:
        next_date = monthly_returns.index[monthly_returns.index.get_loc(current_date) + 1]
        history = monthly_returns.loc[:current_date].tail(lookback_months)
        assets = list(history.dropna(axis=1).columns)
        next_returns = monthly_returns.loc[next_date, assets].dropna()
        assets = list(next_returns.index)
        history = history[assets]

        strategy_weights = {
            "Equal Weight": equal_weight_weights(assets),
            "Mean-Variance": mean_variance_weights(history.mean(), history.cov(), max_weight),
        }

        train = features[features.index < current_date]
        predict = features[(features.index == current_date) & (features["Ticker"].isin(assets))]
        if not train.empty and not predict.empty:
            for model_name, model in models.items():
                model.fit(train[FEATURE_COLUMNS], train["Target_Next_Return"])
                predictions = pd.Series(model.predict(predict[FEATURE_COLUMNS]), index=predict["Ticker"]).reindex(assets).dropna()
                if not predictions.empty:
                    strategy_weights[model_name] = normalize_long_only(predictions, max_weight)

        for strategy, weights in strategy_weights.items():
            weights = weights.reindex(assets).fillna(0)
            records.append({"Date": next_date, "Strategy": strategy, "Return": weights @ next_returns})
            for ticker, weight in weights.items():
                weights_records.append({"Date": next_date, "Strategy": strategy, "Ticker": ticker, "Weight": weight})

    if not records:
        return pd.DataFrame(), pd.DataFrame()
    strategy_returns = pd.DataFrame(records).pivot(index="Date", columns="Strategy", values="Return").dropna()
    strategy_weights = pd.DataFrame(weights_records)
    return strategy_returns, strategy_weights


def prepare_classification_data(features):
    dates = features.index.unique().sort_values()
    split_date = dates[int(len(dates) * 0.75)]
    train = features[features.index < split_date]
    test = features[features.index >= split_date]
    return train, test, split_date


def run_classification_models(features):
    if LogisticRegression is None or RandomForestClassifier is None:
        return None

    train, test, split_date = prepare_classification_data(features)
    x_train = train[FEATURE_COLUMNS]
    y_train = train["Target_Up"]
    x_test = test[FEATURE_COLUMNS]
    y_test = test["Target_Up"]

    models = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

    results = []
    probabilities = {}
    predictions = {}
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        fitted_models[model_name] = model
        probabilities[model_name] = prob
        predictions[model_name] = pred
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, pred),
                "ROC AUC / C-Statistic": roc_auc_score(y_test, prob),
            }
        )

    return {
        "split_date": split_date,
        "x_test": x_test,
        "y_test": y_test,
        "results": pd.DataFrame(results).set_index("Model"),
        "probabilities": probabilities,
        "predictions": predictions,
        "models": fitted_models,
    }


def calculate_information_value(data, feature, target, bins=5):
    temp = data[[feature, target]].dropna().copy()
    temp["Bucket"] = pd.qcut(temp[feature].rank(method="first"), q=bins, duplicates="drop")
    grouped = temp.groupby("Bucket")[target].agg(["count", "sum"])
    grouped["good"] = grouped["sum"]
    grouped["bad"] = grouped["count"] - grouped["sum"]
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    grouped["dist_good"] = grouped["good"] / total_good
    grouped["dist_bad"] = grouped["bad"] / total_bad
    grouped["woe"] = np.log((grouped["dist_good"] + 1e-6) / (grouped["dist_bad"] + 1e-6))
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    return grouped["iv"].sum()


def weighted_fundamentals(fundamentals, weights):
    fields = ["marketCap", "trailingPE", "forwardPE", "priceToBook", "dividendYield", "beta", "profitMargins", "returnOnEquity", "debtToEquity"]
    output = {}
    for field in fields:
        if field not in fundamentals:
            output[field] = np.nan
            continue
        values = pd.to_numeric(fundamentals[field], errors="coerce").dropna()
        aligned_weights = weights.reindex(values.index).dropna()
        if values.empty or aligned_weights.sum() == 0:
            output[field] = np.nan
        else:
            aligned_weights = aligned_weights / aligned_weights.sum()
            output[field] = (values * aligned_weights).sum()
    return pd.Series(output, name="Portfolio Weighted Fundamental")


def technical_signal(technical, short_ma, long_ma, rsi_period):
    latest = technical.dropna().iloc[-1]
    score = 0
    reasons = []

    if latest["Close"] > latest[f"SMA_{short_ma}"] > latest[f"SMA_{long_ma}"]:
        score += 2
        reasons.append("price above short and long moving averages")
    elif latest["Close"] < latest[f"SMA_{short_ma}"] < latest[f"SMA_{long_ma}"]:
        score -= 2
        reasons.append("price below short and long moving averages")

    if latest[f"RSI_{rsi_period}"] < 30:
        score += 1
        reasons.append("RSI indicates oversold conditions")
    elif latest[f"RSI_{rsi_period}"] > 70:
        score -= 1
        reasons.append("RSI indicates overbought conditions")

    if latest["MACD"] > latest["MACD_Signal"]:
        score += 1
        reasons.append("MACD is above signal line")
    elif latest["MACD"] < latest["MACD_Signal"]:
        score -= 1
        reasons.append("MACD is below signal line")

    if score >= 3:
        label = "Strong Buy"
    elif score == 2:
        label = "Buy"
    elif score <= -3:
        label = "Strong Sell"
    elif score == -2:
        label = "Sell"
    else:
        label = "Hold"

    return label, score, "; ".join(reasons)


def portfolio_profit_loss(strategy_returns, investment_amount):
    cumulative = (1 + strategy_returns).cumprod()
    ending_values = cumulative.iloc[-1] * investment_amount
    profit_loss = ending_values - investment_amount
    output = pd.DataFrame(
        {
            "Initial Investment": investment_amount,
            "Ending Value": ending_values,
            "Profit / Loss": profit_loss,
            "Total Return": profit_loss / investment_amount,
        }
    )
    return output.sort_values("Ending Value", ascending=False)


def portfolio_action_plan(metrics, latest_weights, forecasted_returns, forecasted_weights, technical_signals):
    rows = []
    best_strategy = metrics.index[0]
    rows.append(
        {
            "Area": "Best historical strategy",
            "Finding": f"{best_strategy} has the highest Sharpe Ratio in the backtest.",
            "Possible Action": "Use this strategy as the main benchmark, while checking drawdown risk before increasing allocation.",
        }
    )

    max_drawdown_strategy = metrics["Max Drawdown"].idxmax()
    rows.append(
        {
            "Area": "Downside risk",
            "Finding": f"{max_drawdown_strategy} has the smallest historical drawdown.",
            "Possible Action": "Consider this strategy if the objective is capital protection rather than maximum return.",
        }
    )

    if not forecasted_weights.empty:
        top_forecast_asset = forecasted_weights.idxmax()
        rows.append(
            {
                "Area": "Forward-looking allocation",
                "Finding": f"The forecast model gives the highest suggested weight to {top_forecast_asset}.",
                "Possible Action": "Review this asset's technical and fundamental tabs before changing actual weights.",
            }
        )

    if not forecasted_returns.empty:
        weak_assets = forecasted_returns[forecasted_returns < 0].index.tolist()
        if weak_assets:
            rows.append(
                {
                    "Area": "Negative forecast",
                    "Finding": "Negative next-month forecasts: " + ", ".join(weak_assets),
                    "Possible Action": "Consider reducing exposure or setting tighter risk limits for these assets.",
                }
            )

    sells = technical_signals[technical_signals["Signal"].isin(["Sell", "Strong Sell"])]
    if not sells.empty:
        rows.append(
            {
                "Area": "Technical risk",
                "Finding": "Weak technical signals: " + ", ".join(sells.index.tolist()),
                "Possible Action": "Monitor these assets closely; avoid increasing weight until signals improve.",
            }
        )

    concentrated = latest_weights.max().sort_values(ascending=False)
    if not concentrated.empty and concentrated.iloc[0] > 0.4:
        rows.append(
            {
                "Area": "Concentration",
                "Finding": f"{concentrated.index[0]} has a high maximum allocation in at least one strategy.",
                "Possible Action": "Keep a maximum-weight constraint to avoid over-concentration.",
            }
        )

    return pd.DataFrame(rows)


def render_donut_card(title, percentage, color="#2bbbad"):
    percentage = 0 if pd.isna(percentage) else max(0, min(float(percentage), 100))
    remaining = 100 - percentage
    if plt is None:
        st.metric(title, f"{percentage:.2f}%")
        return

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.pie(
        [percentage, remaining],
        colors=[color, "#e8e8e8"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.30, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{percentage:.2f}%", ha="center", va="center", fontsize=16, fontweight="bold", color=color)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    st.pyplot(fig, use_container_width=False)


def render_rating_gauge(title, score, label, left_label="Strong sell", right_label="Strong buy"):
    if plt is None or pd.isna(score):
        st.metric(title, label)
        return

    score = max(-2, min(float(score), 2))
    angle = np.deg2rad(180 - ((score + 2) / 4) * 180)
    needle_x = 0.78 * np.cos(angle)
    needle_y = 0.78 * np.sin(angle)

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    segments = [
        (-2, -1.2, "#ef476f"),
        (-1.2, -0.35, "#c77dff"),
        (-0.35, 0.35, "#d8d8d8"),
        (0.35, 1.2, "#8ecae6"),
        (1.2, 2, "#4361ee"),
    ]
    for start, end, color in segments:
        theta = np.linspace(180 - ((start + 2) / 4) * 180, 180 - ((end + 2) / 4) * 180, 60)
        x = np.cos(np.deg2rad(theta))
        y = np.sin(np.deg2rad(theta))
        ax.plot(x, y, linewidth=6, color=color, solid_capstyle="butt")

    ax.plot([0, needle_x], [0, needle_y], color="#1f1f1f", linewidth=1.6)
    ax.scatter([0], [0], s=22, color="#111111", zorder=5)

    ax.text(-1.02, -0.02, left_label, ha="center", va="center", fontsize=7, color="#999999", fontweight="bold")
    ax.text(1.02, -0.02, right_label, ha="center", va="center", fontsize=7, color="#999999", fontweight="bold")
    ax.text(0, 1.10, "Neutral", ha="center", va="center", fontsize=8, color="#999999", fontweight="bold")
    ax.text(-0.73, 0.58, "Sell", ha="center", va="center", fontsize=7, color="#aaaaaa", fontweight="bold")
    ax.text(0.73, 0.58, "Buy", ha="center", va="center", fontsize=7, color="#aaaaaa", fontweight="bold")
    ax.text(0, -0.34, label, ha="center", va="center", fontsize=13, fontweight="bold", color="#111111")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.25)
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)


def render_signal_card(title, score, label, subtitle=""):
    if pd.isna(score):
        marker = 50
    else:
        marker = ((max(-2, min(float(score), 2)) + 2) / 4) * 100
    safe_subtitle = subtitle or ""
    st.markdown(
        f"""
        <div style="border:1px solid #e6e8ee;border-radius:14px;padding:18px 18px 14px 18px;margin-bottom:10px;">
          <div style="font-size:15px;color:#666;margin-bottom:8px;">{title}</div>
          <div style="font-size:30px;font-weight:700;margin-bottom:12px;">{label}</div>
          <div style="height:12px;border-radius:999px;background:linear-gradient(90deg,#ef476f 0%,#f4a261 25%,#dddddd 50%,#7bd88f 75%,#2bbbad 100%);position:relative;">
            <div style="position:absolute;left:calc({marker:.1f}% - 7px);top:-5px;width:18px;height:18px;border-radius:50%;background:#111;border:3px solid white;box-shadow:0 1px 4px rgba(0,0,0,.25);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-top:7px;">
            <span>Strong sell</span><span>Neutral</span><span>Strong buy</span>
          </div>
          <div style="font-size:13px;color:#777;margin-top:10px;">{safe_subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_value(value, formatter=None):
    if pd.isna(value) or value is None:
        return "Not available"
    if formatter:
        return formatter(value)
    return value


def technical_score_to_gauge(signal):
    mapping = {
        "Strong Sell": -2,
        "Sell": -1,
        "Hold": 0,
        "Buy": 1,
        "Strong Buy": 2,
    }
    return mapping.get(signal, np.nan)


def analyst_rating_to_gauge(recommendation, mean_rating):
    text_mapping = {
        "strong_buy": (2, "Strong buy"),
        "buy": (1, "Buy"),
        "hold": (0, "Hold"),
        "underperform": (-1, "Sell"),
        "sell": (-2, "Strong sell"),
    }
    if isinstance(recommendation, str) and recommendation.lower() in text_mapping:
        return text_mapping[recommendation.lower()]

    if pd.isna(mean_rating):
        return np.nan, "N/A"
    mean_rating = float(mean_rating)
    if mean_rating <= 1.5:
        return 2, "Strong buy"
    if mean_rating <= 2.5:
        return 1, "Buy"
    if mean_rating <= 3.5:
        return 0, "Hold"
    if mean_rating <= 4.5:
        return -1, "Sell"
    return -2, "Strong sell"


def investor_view_from_signal(signal):
    mapping = {
        "Strong Buy": "Positive setup: technical indicators are strongly supportive.",
        "Buy": "Improving setup: technical indicators are mostly positive.",
        "Hold": "Neutral setup: no clear technical direction yet.",
        "Sell": "Caution: technical indicators are weakening.",
        "Strong Sell": "High caution: technical indicators are strongly negative.",
    }
    return mapping.get(signal, "Not enough technical data.")


def format_percent(value):
    if pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) <= 1:
        value *= 100
    return f"{value:.2f}%"


def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def format_date(value):
    if pd.isna(value):
        return "N/A"
    try:
        return pd.to_datetime(value).strftime("%b %d, %Y")
    except Exception:
        return str(value)


st.title("Portfolio Optimization Platform")
st.caption("Analyze any Yahoo Finance tickers, optimize a portfolio, and compare traditional and ML models.")

with st.sidebar:
    st.header("Portfolio Inputs")
    ticker_state_key = "portfolio_ticker_input_v2"
    if st.session_state.get(ticker_state_key, OLD_DEFAULT_TICKER_TEXT) == OLD_DEFAULT_TICKER_TEXT:
        st.session_state[ticker_state_key] = DEFAULT_TICKER_TEXT
    if st.button("Reset to default tickers"):
        st.session_state[ticker_state_key] = DEFAULT_TICKER_TEXT
    ticker_text = st.text_area(
        "Yahoo Finance tickers",
        value=DEFAULT_TICKER_TEXT,
        height=90,
        key=ticker_state_key,
    )
    tickers = list(dict.fromkeys([ticker.strip().upper() for ticker in ticker_text.replace("\n", ",").split(",") if ticker.strip()]))
    start_date = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    end_date = st.date_input("End date", pd.Timestamp.today())
    investment_amount = st.number_input("Investment amount ($)", min_value=100.0, value=10000.0, step=500.0)
    lookback_months = st.slider("Training lookback months", 24, 60, 36)
    max_weight = st.slider("Maximum weight per asset", 0.10, 0.80, 0.35, 0.05)
    st.header("Speed Settings")
    speed_mode = st.selectbox("Analysis speed", ["Fast", "Balanced", "Full"], index=0)
    include_ml_portfolios = st.checkbox("Run ML portfolio strategies", value=speed_mode != "Fast")
    load_asset_intelligence = st.checkbox("Load dividends, analyst data, profile, and news", value=False)
    run_sector_comparison = st.checkbox("Run US sector comparison", value=False)
    run_forecasting = st.checkbox("Run forecasting model", value=False)
    run_ml_diagnostics = st.checkbox("Run ROC/confusion/feature diagnostics", value=False)
    run_deep_learning = st.checkbox("Run deep learning model", value=False)
    estimator_map = {"Fast": 60, "Balanced": 120, "Full": 250}
    model_estimators = estimator_map[speed_mode]
    st.header("Technical Settings")
    short_ma = st.slider("Short moving average", 5, 100, 20)
    long_ma = st.slider("Long moving average", 20, 250, 50)
    rsi_period = st.slider("RSI period", 5, 50, 14)
    st.header("Forecasting")
    forecast_model = st.selectbox("Forecast model", ["Random Forest", "XGBoost"])
    forecast_horizon = st.slider("Forecast horizon - months", 1, 12, 3)
    run = st.button("Run full analysis", type="primary")

if not run:
    st.info("Enter any Yahoo Finance tickers, then click Run full analysis.")
    st.stop()

if len(tickers) < 2:
    st.error("Please enter at least two valid tickers.")
    st.stop()

with st.spinner("Downloading Yahoo Finance data and running the platform..."):
    prices, volume = download_market_data(tickers, start_date, end_date)
    if prices.empty or len(prices.columns) < 2:
        st.error("Not enough valid price data was returned. Check the ticker symbols and date range.")
        st.stop()
    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_tickers:
        st.warning(f"These symbols were not returned by Yahoo Finance: {', '.join(missing_tickers)}")

    valid_tickers = list(prices.columns)
    monthly_returns = prices.resample("ME").last().pct_change().dropna(how="all")
    features = make_features(monthly_returns)
    strategy_returns, strategy_weights = run_portfolio_backtest(
        monthly_returns,
        lookback_months,
        max_weight,
        include_ml=include_ml_portfolios,
        n_estimators=model_estimators,
    )
    if load_asset_intelligence:
        fundamentals = download_fundamentals(valid_tickers)
        asset_intelligence, dividends, earnings, news = download_asset_intelligence(valid_tickers)
    else:
        fundamentals = pd.DataFrame(index=valid_tickers)
        asset_intelligence = pd.DataFrame(index=valid_tickers)
        dividends = pd.DataFrame()
        earnings = pd.DataFrame()
        news = pd.DataFrame()

if strategy_returns.empty:
    st.error("Backtest did not produce results. Try a longer date range or a smaller lookback period.")
    st.stop()

cumulative_returns = (1 + strategy_returns).cumprod()
portfolio_value = cumulative_returns * investment_amount
metrics = strategy_returns.apply(performance_metrics).T.sort_values("Sharpe Ratio", ascending=False)
latest_date = strategy_weights["Date"].max()
latest_weights = strategy_weights[strategy_weights["Date"] == latest_date].pivot(index="Ticker", columns="Strategy", values="Weight").fillna(0)
classification_output = run_classification_models(features) if run_ml_diagnostics else None
if run_forecasting:
    forecasted_returns, forecasted_weights = forecast_next_returns(
        monthly_returns,
        forecast_model,
        max_weight,
        n_estimators=model_estimators,
    )
else:
    forecasted_returns = pd.Series(dtype=float)
    forecasted_weights = pd.Series(dtype=float)
forecasted_portfolio_return = np.nan
forecasted_value_path = pd.Series(dtype=float)
if not forecasted_returns.empty and not forecasted_weights.empty:
    forecasted_portfolio_return = float(forecasted_weights @ forecasted_returns.reindex(forecasted_weights.index))
    forecasted_dates = pd.date_range(monthly_returns.index.max(), periods=forecast_horizon + 1, freq="ME")[1:]
    forecasted_value_path = pd.Series(
        [investment_amount * ((1 + forecasted_portfolio_return) ** step) for step in range(1, forecast_horizon + 1)],
        index=forecasted_dates,
        name="Forecasted Portfolio Value",
    )

technical_signal_rows = []
for ticker in valid_tickers:
    try:
        ticker_technical = add_technical_indicators(prices[ticker], short_ma, long_ma, rsi_period)
        signal, score, reason = technical_signal(ticker_technical, short_ma, long_ma, rsi_period)
        technical_signal_rows.append({
            "Ticker": ticker,
            "Signal": signal,
            "Investor View": investor_view_from_signal(signal),
            "Reason": reason,
            "_Score": score,
        })
    except Exception as exc:
        technical_signal_rows.append({
            "Ticker": ticker,
            "Signal": "N/A",
            "Investor View": "Not enough technical data.",
            "Reason": str(exc),
            "_Score": np.nan,
        })
technical_signals = pd.DataFrame(technical_signal_rows).set_index("Ticker")
profit_loss_table = portfolio_profit_loss(strategy_returns, investment_amount)
action_plan = portfolio_action_plan(metrics, latest_weights, forecasted_returns, forecasted_weights, technical_signals)

metric_cols = st.columns(4)
metric_cols[0].metric("Valid assets", len(valid_tickers))
metric_cols[1].metric("Monthly observations", len(monthly_returns))
metric_cols[2].metric("Best Sharpe strategy", metrics.index[0])
metric_cols[3].metric("Best Sharpe", f"{metrics.iloc[0]['Sharpe Ratio']:.3f}")

tabs = st.tabs(
    [
        "Overview",
        "Statistics",
        "Market Comparison",
        "Technical Analysis",
        "Asset Intelligence",
        "Fundamentals",
        "Optimization",
        "Portfolio Plan",
        "Forecasting",
        "ML Diagnostics",
        "Volume",
        "Downloads",
    ]
)

with tabs[0]:
    st.subheader("Adjusted Closing Prices")
    st.line_chart(prices)
    st.subheader("Portfolio Cumulative Performance")
    st.line_chart(cumulative_returns)
    st.subheader(f"Portfolio Value Growth for ${investment_amount:,.0f}")
    st.line_chart(portfolio_value)

with tabs[1]:
    st.subheader("Descriptive Statistics")
    stats = monthly_returns.describe().T
    stats["Skewness"] = monthly_returns.skew()
    stats["Kurtosis"] = monthly_returns.kurtosis()
    stats["Annualized Return"] = (1 + monthly_returns).prod() ** (12 / len(monthly_returns)) - 1
    stats["Annualized Volatility"] = monthly_returns.std() * np.sqrt(12)
    st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Outliers")
    outliers = detect_outliers(monthly_returns)
    st.dataframe(outliers, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = monthly_returns.corr()
    if sns is not None and plt is not None:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

with tabs[2]:
    st.subheader("Comparison with US Market and Sectors")
    if not run_sector_comparison:
        st.info("Turn on 'Run US sector comparison' in the sidebar to load this analysis.")
    else:
        sector_prices, _ = download_market_data(SECTOR_TICKERS, start_date, end_date)
        sector_returns = sector_prices.resample("ME").last().pct_change().dropna(how="all")
        st.line_chart((1 + sector_returns).cumprod())

        aligned_portfolio = strategy_returns["Equal Weight"].reindex(sector_returns.index).dropna()
        aligned_sectors = sector_returns.reindex(aligned_portfolio.index).dropna()
        aligned_portfolio = aligned_portfolio.reindex(aligned_sectors.index)
        sector_corr = aligned_sectors.corrwith(aligned_portfolio).sort_values(ascending=False)
        st.dataframe(sector_corr.to_frame("Correlation with Equal-Weight Portfolio").style.format("{:.4f}"), use_container_width=True)

with tabs[3]:
    selected_ticker = st.selectbox("Choose asset for technical analysis", valid_tickers)
    technical = add_technical_indicators(prices[selected_ticker], short_ma, long_ma, rsi_period)
    st.subheader(f"Technical Indicators - {selected_ticker}")
    st.dataframe(technical.tail(10).style.format("{:.4f}"), use_container_width=True)
    st.line_chart(technical[["Close", f"SMA_{short_ma}", f"SMA_{long_ma}", f"EMA_{short_ma}", f"EMA_{long_ma}"]])
    st.line_chart(technical[[f"RSI_{rsi_period}"]])
    st.line_chart(technical[["MACD", "MACD_Signal"]])

    st.subheader("Technical Analysis for the Portfolio")
    portfolio_index = cumulative_returns["Equal Weight"]
    portfolio_technical = add_technical_indicators(portfolio_index, short_ma, long_ma, rsi_period)
    st.line_chart(portfolio_technical[["Close", f"SMA_{short_ma}", f"SMA_{long_ma}", f"EMA_{short_ma}", f"EMA_{long_ma}"]])
    st.line_chart(portfolio_technical[[f"RSI_{rsi_period}"]])
    st.dataframe(portfolio_technical.tail(10).style.format("{:.4f}"), use_container_width=True)

with tabs[4]:
    if not load_asset_intelligence:
        st.info("Turn on 'Load dividends, analyst data, profile, and news' in the sidebar to load this section.")
    selected_asset = st.selectbox("Choose asset profile", valid_tickers)
    selected_info = asset_intelligence.loc[selected_asset] if selected_asset in asset_intelligence.index else pd.Series(dtype=float)
    st.subheader(f"Profile and Analyst View - {selected_asset}")

    fallback_price = prices[selected_asset].dropna().iloc[-1] if selected_asset in prices.columns and not prices[selected_asset].dropna().empty else np.nan
    if pd.isna(selected_info.get("Current Price", np.nan)):
        selected_info["Current Price"] = fallback_price

    name = display_value(selected_info.get("Name", selected_asset))
    quote_type = display_value(selected_info.get("Quote Type", ""))
    sector = display_value(selected_info.get("Sector", ""))
    industry = display_value(selected_info.get("Industry", ""))
    website = selected_info.get("Website", np.nan)

    st.markdown(f"### {name}")
    subtitle_parts = [str(part) for part in [quote_type, sector, industry] if str(part) not in ["", "Not available", "nan", "None"]]
    if subtitle_parts:
        st.caption(" | ".join(subtitle_parts))
    if isinstance(website, str) and website:
        st.markdown(f"[Company / fund website]({website})")

    pcols = st.columns(5)
    pcols[0].metric("Current Price", display_value(selected_info.get("Current Price"), lambda x: f"{float(x):,.2f}"))
    pcols[1].metric("Beta", display_value(selected_info.get("Beta"), lambda x: f"{float(x):.2f}"))
    pcols[2].metric("Market Cap", display_value(selected_info.get("Market Cap"), lambda x: f"{float(x):,.0f}"))
    pcols[3].metric("Dividend Rate", display_value(selected_info.get("Dividend Rate"), lambda x: f"{float(x):.3f}"))
    pcols[4].metric("P/B", display_value(selected_info.get("Price To Book"), lambda x: f"{float(x):.2f}"))

    with st.expander("Show full profile fields"):
        profile_cols = [
            "Name",
            "Quote Type",
            "Sector",
            "Industry",
            "Website",
            "Beta",
            "Market Cap",
            "Dividend Yield",
            "Dividend Rate",
            "Payout Ratio",
            "Trailing EPS",
            "Forward EPS",
            "Price To Book",
            "Recommendation",
            "Mean Analyst Rating",
            "Target Mean Price",
            "Current Price",
        ]
        available_profile_cols = [col for col in profile_cols if col in asset_intelligence.columns]
        profile_display = asset_intelligence.loc[[selected_asset], available_profile_cols].replace({None: "Not available"})
        st.dataframe(profile_display, use_container_width=True)

    if selected_info.get("Quote Type", "") == "ETF":
        st.caption("This asset is an ETF, so Yahoo Finance may not provide company-style fields such as analyst rating, EPS, sector, industry, or payout ratio.")

    gauge_left, gauge_right = st.columns([1, 1])
    selected_signal = technical_signals.loc[selected_asset, "Signal"] if selected_asset in technical_signals.index else "N/A"
    with gauge_left:
        reason = technical_signals.loc[selected_asset, "Reason"] if selected_asset in technical_signals.index else ""
        render_signal_card("Technicals", technical_score_to_gauge(selected_signal), selected_signal, reason)

    recommendation = selected_info.get("Recommendation", np.nan)
    mean_rating = selected_info.get("Mean Analyst Rating", np.nan)
    analyst_score, analyst_label = analyst_rating_to_gauge(recommendation, mean_rating)
    with gauge_right:
        render_signal_card("Analyst rating", analyst_score, analyst_label)

        target_price = selected_info.get("Target Mean Price", np.nan)
        current_price = selected_info.get("Current Price", np.nan)
        if not pd.isna(target_price) and not pd.isna(current_price) and float(current_price) != 0:
            upside = (float(target_price) / float(current_price)) - 1
            st.metric("1 year price target", f"{float(target_price):,.2f}", f"{upside:.2%}")
        else:
            st.metric("1 year price target", "N/A")

    with st.expander("Show technical signals for all assets"):
        display_signals = technical_signals.drop(columns=["_Score"], errors="ignore")
        st.dataframe(display_signals, use_container_width=True)

    st.subheader("Dividends")
    selected_dividends = dividends[dividends["Ticker"] == selected_asset] if not dividends.empty else pd.DataFrame()
    payout_ratio = selected_info.get("Payout Ratio", np.nan)
    dividend_yield = selected_info.get("Dividend Yield", np.nan)
    computed_dividend_yield = np.nan
    latest_dividend_value = np.nan
    latest_ex_date = np.nan
    last_payment_date = selected_info.get("Ex-Dividend Date", np.nan)

    if selected_dividends.empty:
        st.info("No recent dividend records were returned for this asset.")
    else:
        selected_dividends = selected_dividends.copy()
        selected_dividends["Date"] = pd.to_datetime(selected_dividends["Date"]).dt.tz_localize(None)
        selected_dividends = selected_dividends.sort_values("Date")
        latest_dividend_value = selected_dividends.iloc[-1]["Dividend"]
        latest_ex_date = selected_dividends.iloc[-1]["Date"]
        latest_price = selected_info.get("Current Price", np.nan)
        if pd.isna(latest_price):
            latest_price = prices[selected_asset].dropna().iloc[-1] if selected_asset in prices.columns else np.nan
        price_last_date = prices.index.max()
        trailing_cutoff = pd.to_datetime(price_last_date).tz_localize(None) - pd.DateOffset(years=1)
        trailing_dividends = selected_dividends[selected_dividends["Date"] >= trailing_cutoff]["Dividend"].sum()
        if trailing_dividends == 0 and len(selected_dividends) > 0:
            trailing_dividends = selected_dividends.tail(min(4, len(selected_dividends)))["Dividend"].sum()
        if not pd.isna(latest_price) and latest_price != 0:
            computed_dividend_yield = trailing_dividends / latest_price
        display_dividend_yield = dividend_yield if not pd.isna(dividend_yield) else computed_dividend_yield

        card_left, card_right = st.columns([1, 1.25])
        with card_left:
            donut_value = payout_ratio
            if pd.isna(donut_value):
                donut_value = display_dividend_yield
            if not pd.isna(donut_value) and abs(float(donut_value)) <= 1:
                donut_value = float(donut_value) * 100
            render_donut_card("Dividends", donut_value)
            st.caption("Teal shows payout ratio when available; otherwise calculated dividend yield TTM.")

        with card_right:
            st.markdown("#### Dividend Snapshot")
            st.metric("Dividend yield TTM", format_percent(display_dividend_yield))
            if pd.isna(dividend_yield) and not pd.isna(computed_dividend_yield):
                st.caption("Calculated as dividends paid over the last 12 months divided by the latest price.")
            st.metric("Payout ratio TTM", format_percent(payout_ratio))
            st.metric("Last payment", format_number(latest_dividend_value))
            st.metric("Last ex-dividend date", format_date(latest_ex_date))
            st.metric("Yahoo ex-dividend field", format_date(last_payment_date))

        dividend_summary = selected_dividends["Dividend"].agg(["count", "sum", "mean", "max"])
        div_cols = st.columns(4)
        div_cols[0].metric("Payments shown", f"{dividend_summary['count']:.0f}")
        div_cols[1].metric("Total dividends", f"{dividend_summary['sum']:.3f}")
        div_cols[2].metric("Average dividend", f"{dividend_summary['mean']:.3f}")
        div_cols[3].metric("Largest dividend", f"{dividend_summary['max']:.3f}")

        st.bar_chart(selected_dividends.set_index("Date")["Dividend"])

        if not dividends.empty:
            dividend_by_asset = dividends.groupby("Ticker")["Dividend"].sum().sort_values(ascending=False)
            st.subheader("Dividend Comparison Across Selected Assets")
            st.bar_chart(dividend_by_asset)

        with st.expander("Show dividend table"):
            st.dataframe(selected_dividends.sort_values("Date", ascending=False), use_container_width=True)

    st.subheader("Earnings")
    selected_earnings = earnings[earnings["Ticker"] == selected_asset] if not earnings.empty else pd.DataFrame()
    if selected_earnings.empty:
        st.info("No earnings calendar data was returned for this asset.")
    else:
        st.dataframe(selected_earnings, use_container_width=True)

    st.subheader("Latest News")
    selected_news = news[news["Ticker"] == selected_asset] if not news.empty else pd.DataFrame()
    if selected_news.empty:
        st.info("No recent news was returned by Yahoo Finance for this asset.")
    else:
        for _, item in selected_news.iterrows():
            title = item.get("Title") or "News item"
            url = item.get("URL")
            provider = item.get("Provider") or ""
            if isinstance(url, str) and url:
                st.markdown(f"- [{title}]({url}) {provider}")
            else:
                st.markdown(f"- {title} {provider}")

with tabs[5]:
    st.subheader("Fundamental Analysis for Each Asset")
    if not load_asset_intelligence:
        st.info("Turn on 'Load dividends, analyst data, profile, and news' in the sidebar to load fundamentals.")
    st.dataframe(fundamentals, use_container_width=True)
    st.subheader("Portfolio Weighted Fundamentals")
    equal_weights = equal_weight_weights(valid_tickers)
    st.dataframe(weighted_fundamentals(fundamentals, equal_weights).to_frame().style.format("{:.4f}"), use_container_width=True)

with tabs[6]:
    st.subheader("Performance Metrics")
    st.dataframe(metrics.style.format("{:.4f}"), use_container_width=True)
    st.subheader(f"Latest Portfolio Weights - {latest_date.date()}")
    st.dataframe(latest_weights.style.format("{:.2%}"), use_container_width=True)
    st.bar_chart(latest_weights)

with tabs[7]:
    st.subheader("Portfolio Profit / Loss")
    st.dataframe(profit_loss_table.style.format({
        "Initial Investment": "${:,.2f}",
        "Ending Value": "${:,.2f}",
        "Profit / Loss": "${:,.2f}",
        "Total Return": "{:.2%}",
    }), use_container_width=True)

    st.subheader("Portfolio Action Plan")
    st.write("These are rule-based planning notes generated from the backtest, forecast, technical signals, and optimization outputs. They are not financial advice.")
    st.dataframe(action_plan, use_container_width=True)

    st.subheader("Suggested Forward Weights")
    if forecasted_weights.empty:
        st.warning("No forecast-based weights are available.")
    else:
        plan_weights = pd.DataFrame({
            "Forecast Weight": forecasted_weights,
            "Dollar Allocation": forecasted_weights * investment_amount,
        })
        st.dataframe(plan_weights.style.format({"Forecast Weight": "{:.2%}", "Dollar Allocation": "${:,.2f}"}), use_container_width=True)

with tabs[8]:
    st.subheader(f"Forecasting with {forecast_model}")
    st.write("The forecast estimates next-month returns using the latest available technical features.")
    if not run_forecasting:
        st.info("Turn on 'Run forecasting model' in the sidebar to load this section.")
    elif forecasted_returns.empty or forecasted_weights.empty:
        st.warning("Forecasting is not available. Check that the selected ML package is installed and there is enough data.")
    else:
        forecast_table = pd.DataFrame(
            {
                "Forecasted Next-Month Return": forecasted_returns,
                "Suggested Forecast Weight": forecasted_weights,
            }
        ).fillna(0)
        st.dataframe(forecast_table.style.format("{:.4f}"), use_container_width=True)
        st.metric("Forecasted monthly portfolio return", f"{forecasted_portfolio_return:.2%}")
        st.metric(
            f"Forecasted value after {forecast_horizon} months",
            f"${forecasted_value_path.iloc[-1]:,.2f}" if not forecasted_value_path.empty else "N/A",
        )
        st.subheader("Forecasted Portfolio Value Path")
        st.line_chart(forecasted_value_path)

with tabs[9]:
    st.subheader("Classification Dataset")
    st.write("Target: whether next-month return is positive.")

    if not run_ml_diagnostics:
        st.info("Turn on 'Run ROC/confusion/feature diagnostics' in the sidebar to load this section.")
    elif classification_output is None:
        st.warning("Install scikit-learn to enable ML diagnostics.")
    else:
        st.write(f"Train/test split date: {classification_output['split_date'].date()}")
        st.subheader("Model Accuracy and ROC AUC / C-Statistic")
        st.dataframe(classification_output["results"].style.format("{:.4f}"), use_container_width=True)

        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model_name, prob in classification_output["probabilities"].items():
                fpr, tpr, _ = roc_curve(classification_output["y_test"], prob)
                auc_value = roc_auc_score(classification_output["y_test"], prob)
                ax.plot(fpr, tpr, label=f"{model_name} AUC={auc_value:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title("ROC Curve Comparison")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)

        selected_model = st.selectbox("Choose model for confusion matrix", list(classification_output["predictions"].keys()))
        cm = pd.DataFrame(
            confusion_matrix(classification_output["y_test"], classification_output["predictions"][selected_model]),
            index=["Actual Down/Zero", "Actual Up"],
            columns=["Predicted Down/Zero", "Predicted Up"],
        )
        st.dataframe(cm, use_container_width=True)
        st.text(classification_report(classification_output["y_test"], classification_output["predictions"][selected_model]))

        st.subheader("Feature Importance")
        importance_rows = []
        for model_name, model in classification_output["models"].items():
            if model_name == "Logistic Regression":
                coefs = model.named_steps["model"].coef_[0]
                for feature, value in zip(FEATURE_COLUMNS, np.abs(coefs)):
                    importance_rows.append({"Model": model_name, "Feature": feature, "Importance": value})
            elif hasattr(model, "feature_importances_"):
                for feature, value in zip(FEATURE_COLUMNS, model.feature_importances_):
                    importance_rows.append({"Model": model_name, "Feature": feature, "Importance": value})
        importance = pd.DataFrame(importance_rows)
        st.dataframe(importance.pivot(index="Feature", columns="Model", values="Importance").fillna(0).style.format("{:.4f}"), use_container_width=True)

        if sm is not None:
            st.subheader("OLS, Logistic, AIC, T-Statistics, P-Values")
            ols_x = sm.add_constant(features[FEATURE_COLUMNS])
            ols_y = features["Target_Next_Return"]
            ols_model = sm.OLS(ols_y, ols_x).fit()
            ols_table = pd.DataFrame(
                {
                    "Coefficient": ols_model.params,
                    "T-Statistic": ols_model.tvalues,
                    "P-Value": ols_model.pvalues,
                }
            )
            st.write(f"OLS AIC: {ols_model.aic:.4f}")
            st.dataframe(ols_table.style.format("{:.4f}"), use_container_width=True)

            logit_x = sm.add_constant(features[FEATURE_COLUMNS])
            logit_y = features["Target_Up"]
            try:
                logit_model = sm.Logit(logit_y, logit_x).fit(disp=False)
                logit_table = pd.DataFrame(
                    {
                        "Coefficient": logit_model.params,
                        "T/Z-Statistic": logit_model.tvalues,
                        "P-Value": logit_model.pvalues,
                    }
                )
                st.write(f"Logistic AIC: {logit_model.aic:.4f}")
                st.dataframe(logit_table.style.format("{:.4f}"), use_container_width=True)
            except Exception as exc:
                st.warning(f"Logistic regression diagnostics could not be estimated: {exc}")

            st.subheader("Information Value and VIF")
            iv = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Information Value": [calculate_information_value(features, feature, "Target_Up") for feature in FEATURE_COLUMNS],
                }
            )
            st.dataframe(iv.sort_values("Information Value", ascending=False).style.format({"Information Value": "{:.4f}"}), use_container_width=True)

            if variance_inflation_factor is not None:
                vif_x = sm.add_constant(features[FEATURE_COLUMNS].dropna())
                vif = pd.DataFrame(
                    {
                        "Variable": vif_x.columns,
                        "VIF": [variance_inflation_factor(vif_x.values, i) for i in range(vif_x.shape[1])],
                    }
                )
                st.dataframe(vif.style.format({"VIF": "{:.4f}"}), use_container_width=True)

        if TimeSeriesSplit is not None:
            st.subheader("Cross Validation")
            cv_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))])
            scores = cross_val_score(cv_model, features[FEATURE_COLUMNS], features["Target_Up"], cv=TimeSeriesSplit(n_splits=5), scoring="roc_auc")
            st.write(f"Logistic Regression Time-Series CV ROC AUC: mean={scores.mean():.4f}, std={scores.std():.4f}")

        st.subheader("Deep Learning - Advanced")
        if not run_deep_learning:
            st.info("Turn on 'Run deep learning model' in the sidebar to train this optional model.")
        elif MLPClassifier is None:
            st.warning("Install scikit-learn to enable the neural-network model.")
        else:
            train, test, _ = prepare_classification_data(features)
            dl_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)),
                ]
            )
            dl_model.fit(train[FEATURE_COLUMNS], train["Target_Up"])
            dl_prob = dl_model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            dl_pred = (dl_prob >= 0.5).astype(int)
            st.write(f"Deep Learning ROC AUC: {roc_auc_score(test['Target_Up'], dl_prob):.4f}")
            st.write(f"Deep Learning Accuracy: {accuracy_score(test['Target_Up'], dl_pred):.4f}")

with tabs[10]:
    st.subheader("Monthly Trading Volume")
    if not volume.empty:
        monthly_volume = volume.resample("ME").sum().dropna(how="all")
        st.line_chart(monthly_volume)
        st.dataframe(monthly_volume.tail(12), use_container_width=True)
    else:
        st.warning("Volume data was not available for the selected assets.")

with tabs[11]:
    st.subheader("Download Results")
    st.download_button("Download performance metrics CSV", metrics.to_csv().encode("utf-8"), "performance_metrics.csv")
    st.download_button("Download strategy returns CSV", strategy_returns.to_csv().encode("utf-8"), "strategy_returns.csv")
    st.download_button("Download latest weights CSV", latest_weights.to_csv().encode("utf-8"), "latest_weights.csv")

missing = []
if RandomForestRegressor is None:
    missing.append("scikit-learn")
if XGBRegressor is None:
    missing.append("xgboost")
if sm is None:
    missing.append("statsmodels")
if sns is None:
    missing.append("seaborn")
if missing:
    st.warning("Some features require missing packages: " + ", ".join(missing))
