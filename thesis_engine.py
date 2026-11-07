"""Chapter 3 research implementation used by the EXISTING Streamlit app.

Prices and forecasts are monthly. Models use a pooled panel; targets are next-
month returns; rebalancing is out-of-sample with expanding training windows.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DEFAULT = ["SPY", "GLD", "TLT", "IEFA", "VNQ", "USO"]
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
SERIES = {
    "real10": "DFII10", "treasury10": "DGS10", "treasury3": "DGS3MO",
    "usd": "DTWEXBGS", "vix": "VIXCLS", "tbill": "DTB3", "cpi": "CPIAUCSL",
    "production": "INDPRO", "unemployment": "UNRATE",
}
TECH = ["RSI14", "BB_PctB", "BB_Width", "Price_SMA20", "SMA20_SMA50",
        "Fib_Nearest", "Return_1M", "Lag_1M", "Lag_2M", "Lag_3M",
        "Momentum_3M", "Momentum_6M", "Volatility_21D"]
FINANCIAL = ["Real10Y", "TermSpread", "USD_Change", "VIX"]
ECONOMIC = ["Inflation_YoY", "IndustrialGrowth_YoY", "Unemployment"]
POLITICAL = ["GPR"]
GROUPS = {
    "Technical": TECH, "Financial": FINANCIAL,
    "Real economy": ECONOMIC, "Political": POLITICAL,
}
FEATURES = TECH + FINANCIAL + ECONOMIC + POLITICAL


def last_complete_month(today=None):
    d = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    return d.replace(day=1) - pd.Timedelta(days=1)


def get_prices(tickers, start, today=None):
    """Use pre-study warmup, but never include the current unfinished month."""
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if len(tickers) < 2:
        raise ValueError("Choose at least two different stock/ETF symbols.")
    start = pd.Timestamp(start)
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today())
    raw = yf.download(tickers, start=(start-pd.DateOffset(months=14)).date(),
                      end=(today+pd.Timedelta(days=1)).date(),
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no prices.")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # Do not fabricate missing prices over longer gaps.
    close = close.sort_index().ffill(limit=2).dropna(how="all")
    monthly = close.resample("ME").last().loc[:last_complete_month(today)]
    candidate = monthly.loc[monthly.index >= start]
    usable = [x for x in tickers if candidate[x].notna().sum() >= 55]
    if len(usable) < 2:
        raise ValueError("Need at least two assets with 55+ complete monthly observations.")
    monthly = monthly[usable].dropna(how="any")
    close = close[usable]
    if len(monthly) < 60:
        raise ValueError("Too few overlapping months; select assets with longer histories.")
    return close, monthly


def technical_panel(daily, monthly):
    mret = monthly.pct_change(fill_method=None)
    panels = []
    for symbol in monthly:
        p = daily[symbol].dropna()
        delta = p.diff()
        up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rsi = 100-(100/(1+up/down.replace(0,np.nan)))
        ma20, ma50 = p.rolling(20).mean(), p.rolling(50).mean()
        sd20 = p.rolling(20).std()
        top, bottom = ma20+2*sd20, ma20-2*sd20
        high63, low63 = p.rolling(63).max(), p.rolling(63).min()
        fib = pd.concat([low63+(high63-low63)*ratio
                         for ratio in (.236,.382,.5,.618,.786)],axis=1)
        d = pd.DataFrame({
            "RSI14": rsi,
            "BB_PctB": (p-bottom)/(top-bottom).replace(0,np.nan),
            "BB_Width": (top-bottom)/ma20,
            "Price_SMA20": p/ma20,
            "SMA20_SMA50": ma20/ma50,
            "Fib_Nearest": fib.sub(p,axis=0).abs().min(axis=1)/p,
            # Daily-data indicator annualized as in Chapter 3, not portfolio volatility.
            "Volatility_21D": p.pct_change(fill_method=None).rolling(21).std()*np.sqrt(252),
        }).resample("ME").last().reindex(monthly.index)
        r = mret[symbol]
        d["Return_1M"] = r
        for n in (1,2,3):
            d[f"Lag_{n}M"] = r.shift(n)
        d["Momentum_3M"] = monthly[symbol]/monthly[symbol].shift(3)-1
        d["Momentum_6M"] = monthly[symbol]/monthly[symbol].shift(6)-1
        d["Target_Next_Return"] = r.shift(-1)
        d["Ticker"] = symbol
        d.index.name = "Date"
        panels.append(d.reset_index())
    return pd.concat(panels,ignore_index=True).sort_values(["Date","Ticker"])


def _fred_csv(code):
    raw = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id="+code)
    datekey = "observation_date" if "observation_date" in raw else "DATE"
    if datekey not in raw or code not in raw:
        raise ValueError(f"FRED {code} returned an unexpected format.")
    vals = pd.to_numeric(raw[code],errors="coerce")
    dates = pd.to_datetime(raw[datekey],errors="coerce")
    return pd.Series(vals.values,index=dates).dropna().sort_index()


def fundamental_economic_factors():
    """Conservative 2-month release lag for CPI/production/unemployment.
    FRED's revised latest series are not true historical point-in-time vintages.
    """
    r = {k:_fred_csv(code) for k,code in SERIES.items()}
    raw = pd.DataFrame()
    for key in SERIES:
        monthly = r[key].resample("ME").mean() if key == "vix" else r[key].resample("ME").last()
        raw[key] = monthly
    f = pd.DataFrame(index=raw.index)
    f["Real10Y"] = raw["real10"].ffill(limit=1)
    f["TermSpread"] = (raw["treasury10"]-raw["treasury3"]).ffill(limit=1)
    f["USD_Change"] = raw["usd"].pct_change(fill_method=None)
    f["VIX"] = raw["vix"]
    f["Inflation_YoY"] = (100*raw["cpi"].pct_change(12,fill_method=None)).shift(2)
    f["IndustrialGrowth_YoY"] = (100*raw["production"].pct_change(12,fill_method=None)).shift(2)
    f["Unemployment"] = raw["unemployment"].shift(2)
    f["RF_Monthly"] = raw["tbill"].ffill(limit=1)/1200
    return f


def geopolitical_factor():
    """Official Caldara-Iacoviello monthly GPR, available after month closes."""
    raw = pd.read_excel(GPR_URL,engine="xlrd")
    # Monthly series supplied by the official source; retain global GPR only.
    cols = {str(k).strip().lower():k for k in raw.columns}
    monthcol = cols.get("month") or cols.get("date")
    valuecol = cols.get("gpr")
    if monthcol is None or valuecol is None:
        raise ValueError("Official GPR Excel layout changed; review column labels.")
    x = raw[monthcol]
    dates = pd.to_datetime(x.astype(str).str.replace(r"\.0$","",regex=True),
                           format="%Y%m",errors="coerce")
    if dates.notna().sum() < len(x)*0.7:
        dates = pd.to_datetime(x,errors="coerce")
    values = pd.to_numeric(raw[valuecol],errors="coerce")
    gpr = pd.Series(values.to_numpy(),index=dates).dropna()
    gpr = gpr.loc[gpr.index.notna()]
    gpr.index = gpr.index.to_period("M").to_timestamp("M")
    gpr = gpr[~gpr.index.duplicated(keep="last")].sort_index()
    if gpr.empty:
        raise ValueError("No usable monthly GPR observations in source file.")
    return gpr.shift(1).rename("GPR")


def full_panel(technical, financial, political):
    macro = financial.join(political,how="left")
    macro.index.name = "Date"
    panel = technical.merge(macro.reset_index(),on="Date",how="left",validate="many_to_one")
    panel = panel.replace([np.inf,-np.inf],np.nan)
    panel = panel.dropna(subset=FEATURES+["RF_Monthly"])
    counts = panel.groupby("Date")["Ticker"].nunique()
    expected = technical["Ticker"].nunique()
    panel = panel.loc[panel["Date"].isin(counts[counts == expected].index)].copy()
    dates = sorted(panel["Date"].unique())
    if len(dates) < 50:
        raise ValueError(f"Only {len(dates)} complete panel months after factor merging.")
    return panel.sort_values(["Date","Ticker"]).reset_index(drop=True)


def make_model(kind,params=None):
    if kind=="Random Forest":
        opts = dict(n_estimators=90,max_depth=7,min_samples_leaf=5,
                    max_features=0.8,random_state=42,n_jobs=-1)
        opts.update(params or {})
        return RandomForestRegressor(**opts)
    opts = dict(n_estimators=130,max_depth=3,learning_rate=0.04,
                min_child_weight=5,subsample=0.8,colsample_bytree=0.85,
                reg_lambda=4,objective="reg:squarederror",random_state=42,n_jobs=2)
    opts.update(params or {})
    return XGBRegressor(**opts)


def tune_params(train, features, kind):
    """TimeSeriesSplit over unique dates, NEVER rows/assets."""
    months = np.array(sorted(train["Date"].unique()))
    grids = ([dict(max_depth=5,min_samples_leaf=5),
              dict(max_depth=8,min_samples_leaf=8)] if kind=="Random Forest"
             else [dict(max_depth=2,learning_rate=0.05),
                   dict(max_depth=3,learning_rate=0.035)])
    n_splits = 3 if len(months)>=40 else 2
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scored = []
    for params in grids:
        scores=[]
        for tr_idx,va_idx in splitter.split(months):
            tr = train[train["Date"].isin(months[tr_idx])]
            va = train[train["Date"].isin(months[va_idx])]
            model = make_model(kind,params)
            model.fit(tr[features],tr["Target_Next_Return"])
            yhat=model.predict(va[features])
            scores.append(float(np.sqrt(mean_squared_error(va["Target_Next_Return"],yhat))))
        scored.append({"Model":kind,"Parameters":str(params),"CV_RMSE":float(np.mean(scores))})
    best = int(np.argmin([x["CV_RMSE"] for x in scored]))
    return grids[best],scored


def max_sharpe_weights(mu,cov,rf):
    """Full investment and nonnegative weights, no maximum cap (Chapter 3)."""
    mu=np.asarray(mu,dtype=float)
    cov=np.asarray(cov,dtype=float)
    n=len(mu)
    cov=(cov+cov.T)/2
    def score(w):
        variance=max(float(w @ cov @ w),1e-12)
        return -(float(w @ mu)-rf)/np.sqrt(variance)
    initial=[np.ones(n)/n] + [np.eye(n)[i] for i in range(n)]
    best=None
    for x0 in initial:
        res=minimize(score,x0,method="SLSQP",bounds=[(0.,1.)]*n,
                     constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}],
                     options={"maxiter":300,"ftol":1e-10})
        if res.success and np.isfinite(res.fun):
            if best is None or res.fun<best.fun:best=res
    w = best.x if best is not None else np.ones(n)/n
    w=np.clip(w,0,1)
    return w/w.sum()


def evaluate_portfolios(monthly, panel):
    """Expanding-window regressors, 36M monthly Ledoit-Wolf risk estimation."""
    tickers=list(monthly.columns)
    # Panel dates refer to the information available at formation month-end.
    # Use one consistent realized monthly return target for every strategy.
    returns=monthly.pct_change(fill_method=None)
    months=sorted(panel.loc[panel["Target_Next_Return"].notna(),"Date"].unique())
    split=int(len(months)*0.70)
    train_months=months[:split]
    test_months=months[split:]
    if len(train_months)<36 or len(test_months)<6:
        raise ValueError("Insufficient chronological training/testing months.")
    initial=panel[panel["Date"].isin(train_months)].dropna(
        subset=["Target_Next_Return"]).copy()
    id_columns=[f"ETF_{t}" for t in tickers[1:]]
    for t in tickers[1:]:
        panel[f"ETF_{t}"]=(panel["Ticker"]==t).astype(int)
        initial[f"ETF_{t}"]=(initial["Ticker"]==t).astype(int)
    feats=FEATURES+id_columns
    tuned, cv=[],[]
    for modelname in ("Random Forest","XGBoost"):
        chosen, rows=tune_params(initial,feats,modelname)
        tuned.append((modelname,chosen))
        cv.extend(rows)
    portfolio_rows=[]
    prediction_rows=[]
    weight_rows=[]
    importance_rows=[]
    for dt in test_months:
        sample=panel[panel["Date"]==dt].set_index("Ticker").reindex(tickers)
        sample["Ticker"]=tickers
        training=panel[(panel["Date"]<dt)&panel["Target_Next_Return"].notna()]
        if len(training)<36*len(tickers):
            continue
        risk_hist=returns.loc[:dt,tickers].tail(36)
        if risk_hist.isna().any().any() or len(risk_hist)!=36:
            continue
        covariance=LedoitWolf().fit(risk_hist.to_numpy()).covariance_
        rf=float(sample["RF_Monthly"].iloc[0])
        mu_hist=risk_hist.mean().to_numpy()
        next_dt=monthly.index[monthly.index.get_loc(dt)+1]
        actual=returns.loc[next_dt,tickers]
        mu_dict={"Mean-Variance":mu_hist}
        for modelname,params in tuned:
            model=make_model(modelname,params)
            model.fit(training[feats],training["Target_Next_Return"])
            pred=model.predict(sample[feats])
            mu_dict[modelname]=pred
            for i,t in enumerate(tickers):
                prediction_rows.append({"FormationDate":dt,"RealizedDate":next_dt,
                   "Ticker":t,"Model":modelname,"Actual":float(actual[t]),
                   "Predicted":float(pred[i])})
            for feature,value in zip(feats,model.feature_importances_):
                importance_rows.append({"Date":dt,"Model":modelname,"Feature":feature,
                                        "Importance":float(value)})
        for strategy,mu in mu_dict.items():
            weights=max_sharpe_weights(mu,covariance,rf)
            realized=float(weights @ actual.reindex(tickers).to_numpy())
            portfolio_rows.append({"Date":next_dt,"FormationDate":dt,
                "Strategy":strategy,"Return":realized,"RiskFree":rf,
                "VIX":sample["VIX"].iloc[0],"GPR":sample["GPR"].iloc[0]})
            for t,w in zip(tickers,weights):
                weight_rows.append({"Date":next_dt,"Strategy":strategy,
                                    "Ticker":t,"Weight":float(w)})
    returns=pd.DataFrame(portfolio_rows)
    if returns.empty:raise ValueError("The walk-forward backtest produced no observations.")
    values=returns.pivot(index="Date",columns="Strategy",values="Return")
    if values.isna().any().any():raise ValueError("Incomplete strategy comparisons; recheck inputs.")
    preds=pd.DataFrame(prediction_rows)
    weights=pd.DataFrame(weight_rows)
    importance=pd.DataFrame(importance_rows)
    dates={"TrainStart":pd.Timestamp(train_months[0]),"TrainEnd":pd.Timestamp(train_months[-1]),
           "TestFormationStart":pd.Timestamp(test_months[0]),"TestFormationEnd":pd.Timestamp(test_months[-1]),
           "TestReturnStart":values.index.min(),"TestReturnEnd":values.index.max()}
    return {"returns":values,"predictions":preds,"weights":weights,
            "importance":importance,"conditions":returns,
            "cv":pd.DataFrame(cv),"dates":dates,
            "parameters":dict(tuned),"panel":panel}


def metrics(result,monthly):
    returns=result["returns"]
    conditions=result["conditions"]
    rf = conditions[conditions["Strategy"]=="Mean-Variance"].set_index("Date")["RiskFree"].reindex(returns.index)
    market = monthly["SPY"].pct_change(fill_method=None).reindex(returns.index) if "SPY" in monthly else None
    rows=[]
    for name in returns:
        x=returns[name].dropna()
        sd=x.std()
        wealth=(1+x).cumprod()
        draw=wealth/wealth.cummax()-1
        row={"Strategy":name,"Cumulative Return":float(wealth.iloc[-1]-1),
             "Mean Monthly Return":float(x.mean()),
             "Monthly Volatility":float(sd),
             "Monthly Sharpe":float((x-rf.reindex(x.index)).mean()/sd) if sd else np.nan,
             "Historical VaR 95%":float(max(0,-x.quantile(.05))),
             "Maximum Drawdown":float(draw.min())}
        if market is not None:
            paired=pd.concat([x,market],axis=1).dropna()
            row["Beta vs SPY"]=float(paired.iloc[:,0].cov(paired.iloc[:,1])/paired.iloc[:,1].var()) if paired.iloc[:,1].var() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


def prediction_errors(pred):
    out=[]
    for (model,ticker),frame in pred.groupby(["Model","Ticker"],dropna=False):
        out.append({"Model":model,"Ticker":ticker,
                    "MAE":mean_absolute_error(frame["Actual"],frame["Predicted"]),
                    "RMSE":np.sqrt(mean_squared_error(frame["Actual"],frame["Predicted"])),
                    "Observations":len(frame)})
    for model,frame in pred.groupby("Model"):
        out.append({"Model":model,"Ticker":"ALL",
                    "MAE":mean_absolute_error(frame["Actual"],frame["Predicted"]),
                    "RMSE":np.sqrt(mean_squared_error(frame["Actual"],frame["Predicted"])),
                    "Observations":len(frame)})
    return pd.DataFrame(out)


def group_importance(importance):
    rows=[]
    for (model,feature),frame in importance.groupby(["Model","Feature"]):
        label=next((k for k,items in GROUPS.items() if feature in items),"Asset identifier")
        rows.append({"Model":model,"Feature":feature,"Group":label,
                     "Importance":float(frame["Importance"].mean())})
    return pd.DataFrame(rows)


def regime_comparison(result):
    condition=result["conditions"].copy()
    train=result["panel"]
    cutoff=result["dates"]["TrainEnd"]
    baseline=train[train["Date"]<=cutoff].drop_duplicates("Date")
    for col in ("VIX","GPR"):
        threshold=float(baseline[col].median())
        condition[col+"_Regime"]=np.where(condition[col]>threshold,"High","Low")
    rows=[]
    for regimecol in ("VIX_Regime","GPR_Regime"):
        for (name,state),group in condition.groupby(["Strategy",regimecol]):
            r=group["Return"]
            sd=r.std()
            rows.append({"Factor":regimecol,"Condition":state,"Strategy":name,
                        "Months":len(group),"Mean Monthly Return":r.mean(),
                        "Monthly Volatility":sd,
                        "Monthly Sharpe":(r-group["RiskFree"]).mean()/sd if sd and len(r)>1 else np.nan})
    return pd.DataFrame(rows)


def current_allocations(monthly, panel, parameters):
    """Formation-month recommendations; targets for the current month are unknown."""
    symbols=list(monthly.columns)
    dt=panel["Date"].max()
    latest=panel.loc[panel["Date"]==dt].set_index("Ticker").reindex(symbols)
    history=monthly.pct_change(fill_method=None).loc[:dt,symbols].tail(36)
    if history.isna().any().any() or len(history)!=36:
        return pd.DataFrame()
    covariance=LedoitWolf().fit(history.to_numpy()).covariance_
    rf=float(latest["RF_Monthly"].iloc[0])
    columns=FEATURES+[f"ETF_{t}" for t in symbols[1:]]
    train=panel.loc[(panel["Date"]<dt)&panel["Target_Next_Return"].notna()].copy()
    scores={"Mean-Variance":history.mean().to_numpy()}
    for modelname,params in parameters.items():
        model=make_model(modelname,params)
        model.fit(train[columns],train["Target_Next_Return"])
        scores[modelname]=model.predict(latest[columns])
    records=[]
    for modelname,pred in scores.items():
        w=max_sharpe_weights(pred,covariance,rf)
        for asset,value in zip(symbols,w):
            records.append({"FormationDate":dt,"Strategy":modelname,
                            "Ticker":asset,"Weight":float(value)})
    return pd.DataFrame(records)
