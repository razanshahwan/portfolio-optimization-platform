# Portfolio Optimization Platform

Streamlit research and educational portfolio comparison tool.

**Existing app URL:** https://portfolio-optimization-platform.streamlit.app/

## Getting started
1. Enter any stock or ETF symbols, separated by commas (or restore the six thesis ETFs).
2. Verify the full company/fund names and choose your comparison goal.
3. Click **Build & compare portfolios**.
4. Compare historical monthly risk and performance, view estimated portfolio allocations, and optionally inspect the research diagnostics.
5. Download Chapter 4 research tables as a ZIP containing CSVs.

## Default research portfolio
SPY, GLD, TLT, IEFA, VNQ, USO. Users may select other stocks or ETFs, provided that sufficient price history is available.

## Research methodology
- Market source: Yahoo Finance, adjusted close, daily data aggregated to **completed months** only.
- Additional factors: Federal Reserve Economic Data (FRED) and Caldara–Iacoviello GPR.
- Four feature groups: technical, financial, real economy, political.
- Machine learning: pooled Random Forest and XGBoost regression, ETF identifiers, time-series training-only CV.
- Benchmark: historical mean–variance portfolio.
- Split: earliest 70% of labeled monthly formation dates for training; most recent 30% for test.
- Rebalancing: monthly walk-forward and expanding training. All strategies use monthly covariance of the last 36 ETF returns estimated via Ledoit–Wolf; long-only and fully invested.
- Evaluation: monthly average return and volatility, cumulative return, monthly Sharpe, beta (when SPY is selected), 95% historical VaR, max drawdown, prediction MAE/RMSE.

## Run locally
```bash
pip install -r requirements.txt
streamlit run portfolio_platform.py
```

The older deployment entry point `portfolio_dashboard.py` now loads `portfolio_platform.py` so the existing site displays the same streamlined interface whichever entry script the deployment uses.

## Reproducibility and interpretation
- The current unfinished calendar month is omitted.
- Some economic reports have release delays. The code applies a conservative two-month lag to selected monthly indicators.
- FRED historical values may be revised, so this version is **not** a point-in-time macro vintage backtest. A publication-date/vintage robustness audit is needed before writing definitive thesis claims.
- Long-only maximum-Sharpe portfolios can be concentrated.
- Backtests omit transaction costs, commissions, taxes and market impact.
- Source availability affects outputs. Errors should be shown rather than substituting fictitious data.
- This is a historical educational comparison, not personalized financial advice.
