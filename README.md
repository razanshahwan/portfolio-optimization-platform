# Dynamic Portfolio Optimization with Machine Learning

This project supports a master's thesis on dynamic portfolio optimization using
machine learning techniques. It compares machine-learning-based rebalanced
portfolios against traditional portfolio baselines.

## Research Goal

Evaluate whether machine learning can improve portfolio performance compared
with traditional optimization methods under changing market conditions.

## Initial Models

- Equal-weight portfolio (`1/N`)
- Mean-variance portfolio
- Machine-learning portfolio using return forecasts

## Evaluation Metrics

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `py` is not available on your machine, install Python 3.11+ or use the
Python interpreter available in your IDE.

## Run

```powershell
python -m src.main
```

The first version downloads daily prices from Yahoo Finance, builds monthly
rebalanced portfolios, and prints a performance comparison.

## Thesis Asset Universe

The initial portfolio universe uses six liquid ETFs across different asset
classes:

- SPY: US equities
- GLD: gold
- TLT: long-term US Treasury bonds
- IEFA: developed international equities
- USO: oil ETF
- VNQ: real estate investment trusts

Note: `TLT` is a defensive fixed-income ETF, but it is not literally risk-free
because long-term bond prices can fluctuate when interest rates change.
