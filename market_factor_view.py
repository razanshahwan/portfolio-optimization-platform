"""Accessible historical market-factor comparisons for the portfolio website.

Compares realized next-month portfolio returns conditional on information
available at formation time. Descriptive evidence, not causal attribution.
"""
import pandas as pd
import streamlit as st

FACTOR_GROUPS = {
    "Technical analysis": {
        "Technical trend (SMA20 / SMA50)": "Uptrend means the average 20-day moving average of the selected assets was above their 50-day average.",
        "Technical RSI14": "Compares stronger versus weaker recent price momentum.",
        "Technical Bollinger %B": "Compares higher and lower positions inside recent price bands.",
        "Technical Fibonacci distance": "Compares distance from Fibonacci retracement levels."
    },
    "Financial conditions": {
        "Financial volatility (VIX)": "High VIX describes higher expected equity-market volatility.",
        "Financial real Treasury rate": "Compares periods of higher and lower real Treasury yields.",
        "Financial yield-curve spread": "Inverted means the longer-term yield was below the short-term yield.",
        "Financial dollar movement": "Compares stronger and weaker U.S. dollar movements."
    },
    "Real economy": {
        "Real economy expansion / slowdown": "Expansion means positive industrial-production growth and no increase in the published unemployment rate.",
        "Real economy inflation": "Compares periods of higher versus lower inflation.",
        "Real economy industrial growth": "Compares periods of stronger versus weaker industrial production growth."
    },
    "Political conditions": {
        "Political geopolitical risk": "The geopolitical risk index (GPR) compares periods of higher and lower international uncertainty."
    }
}

METHODS = ["Mean-Variance", "Random Forest", "XGBoost"]
GROUP_ICONS = {
    "Technical analysis": "📉",
    "Financial conditions": "💵",
    "Real economy": "🏭",
    "Political conditions": "🌐",
}


def render_factor_dashboard(regimes: pd.DataFrame):
    """Show four groups, three strategies, return/risk charts and evidence limits."""
    st.header("How market conditions relate to portfolio performance")
    st.write(
        "Compare what happened to **all three strategies** in the month AFTER "
        "different market conditions were observed. Explore technical trends, "
        "financial markets, the real economy and geopolitical risk."
    )
    if regimes.empty:
        st.warning("Factor comparisons are unavailable for the selected date range.")
        return

    pages = st.tabs([
        GROUP_ICONS[name] + " " + name
        for name in FACTOR_GROUPS
    ])
    for page, (group, factors) in zip(pages, FACTOR_GROUPS.items()):
        with page:
            st.subheader(group)
            factor = st.selectbox(
                "Choose an indicator",
                list(factors.keys()),
                key="factor_" + group.replace(" ", "_"),
            )
            st.caption(factors[factor])
            subset = regimes.loc[regimes["Factor"] == factor].copy()
            if subset.empty:
                st.warning("There are not enough observations for this indicator.")
                continue

            st.markdown("#### Average monthly return")
            returns_chart = subset.pivot(
                index="Strategy",
                columns="Condition",
                values="Mean Monthly Return",
            ).reindex(METHODS) * 100
            st.bar_chart(returns_chart, use_container_width=True)
            st.caption(
                "Each bar shows the average NEXT-month return in percent (%). "
                "The values are not cumulative gains."
            )

            st.markdown("#### Monthly risk (volatility)")
            risks_chart = subset.pivot(
                index="Strategy",
                columns="Condition",
                values="Monthly Volatility",
            ).reindex(METHODS) * 100
            st.bar_chart(risks_chart, use_container_width=True)
            st.caption("Monthly volatility in percent (%). Lower indicates less historical variation.")

            st.markdown("#### Compare the three methods")
            table = subset[[
                "Strategy", "Condition", "Months",
                "Mean Monthly Return", "Monthly Volatility", "Monthly Sharpe",
            ]].sort_values(["Strategy", "Condition"])
            st.dataframe(table.style.format({
                "Mean Monthly Return": "{:.2%}",
                "Monthly Volatility": "{:.2%}",
                "Monthly Sharpe": "{:.2f}",
            }), hide_index=True, use_container_width=True)
            st.caption("Months = number of realized monthly returns in that condition.")

            st.markdown("#### Historical insight")
            conditions = sorted(subset["Condition"].dropna().unique().tolist())
            if conditions:
                focus = st.selectbox(
                    "Look at this condition", conditions,
                    key="regime_" + group.replace(" ", "_"),
                )
                focus_rows = subset.loc[
                    (subset["Condition"] == focus) &
                    subset["Monthly Sharpe"].notna()
                ]
                if not focus_rows.empty and (focus_rows["Months"] >= 5).all():
                    best = focus_rows.loc[focus_rows["Monthly Sharpe"].idxmax()]
                    st.info(
                        f"When **{focus}** was observed, **{best['Strategy']}** had "
                        f"the highest monthly Sharpe ratio in this backtest "
                        f"({best['Monthly Sharpe']:.2f}; {int(best['Months'])} months). "
                        "Compare the return and volatility bars before interpreting that ranking."
                    )
                else:
                    st.info(
                        "This condition has limited observations or missing risk estimates. "
                        "A method ranking is not reliable enough to highlight."
                    )

            st.caption(
                "These are historical associations, NOT proof that any factor caused a "
                "portfolio gain or loss. High/low thresholds use training-period data only. "
                "Every method is compared over the same dates."
            )

    with st.expander("What are the four market-factor groups?"):
        st.write(
            "**Technical:** price trends, RSI, Bollinger Bands, Fibonacci levels, momentum. "
            "**Financial:** interest rates, Treasury yield curve, U.S. dollar and VIX. "
            "**Real economy:** inflation, industrial production and unemployment. "
            "**Political:** the Geopolitical Risk Index."
        )
        st.write(
            "The Mean–Variance method estimates expected returns from historical ETF "
            "returns. Random Forest and XGBoost additionally use the factor indicators "
            "to predict next-month asset returns. All three methods are then evaluated "
            "under the same market conditions."
        )
    st.markdown("**Recommendation:** Use the factor comparisons to assess whether a strategy's "
                "historical results were consistent across different conditions, not to predict "
                "which method will win in the next month.")
