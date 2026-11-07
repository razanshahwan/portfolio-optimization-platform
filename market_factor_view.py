"""Accessible historical market-factor comparisons for the portfolio website.

Compares realized next-month portfolio returns conditional on information
available at formation time. Descriptive evidence, not causal attribution.
"""
import pandas as pd
import streamlit as st
from ui_translation import explanation, arabic_below

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

FACTOR_ARABIC = {
    "Technical trend (SMA20 / SMA50)": "الاتجاه الصاعد يعني أن المتوسط المتحرك لعشرين يومًا أعلى من متوسط خمسين يومًا للأصول المختارة بشكل عام.",
    "Technical RSI14": "مقارنة الفترات ذات قوة الأسعار الأخيرة الأعلى أو الأقل.",
    "Technical Bollinger %B": "مقارنة موقع السعر المرتفع أو المنخفض نسبيًا داخل نطاقات بولينجر.",
    "Technical Fibonacci distance": "مقارنة بُعد السعر عن مستويات تصحيح فيبوناتشي.",
    "Financial volatility (VIX)": "ارتفاع VIX يشير إلى زيادة التقلب المتوقع في سوق الأسهم الأمريكية.",
    "Financial real Treasury rate": "مقارنة الفترات ذات عوائد سندات الخزانة الحقيقية الأعلى والأقل.",
    "Financial yield-curve spread": "انعكاس منحنى العائد يعني أن العائد طويل الأجل أقل من العائد قصير الأجل.",
    "Financial dollar movement": "مقارنة الفترات التي شهدت ارتفاعًا أو انخفاضًا نسبيًا في الدولار الأمريكي.",
    "Real economy expansion / slowdown": "التوسع يعني نموًا صناعيًا موجبًا دون ارتفاع في أحدث معدل بطالة منشور.",
    "Real economy inflation": "مقارنة الفترات التي كان فيها التضخم مرتفعًا أو منخفضًا.",
    "Real economy industrial growth": "مقارنة فترات النمو الصناعي الأقوى والأضعف.",
    "Political geopolitical risk": "مؤشر المخاطر الجيوسياسية GPR يقارن فترات التوتر الجيوسياسي المرتفع والمنخفض."
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
    explanation(
        "Compare what happened to **all three strategies** in the month AFTER "
        "different market conditions were observed. Explore technical trends, "
        "financial markets, the real economy and geopolitical risk.",
        "نقارن أداء المحافظ الثلاث خلال الشهر التالي لظروف سوق مختلفة، "
        "بما يشمل المؤشرات الفنية والمالية والاقتصادية والمخاطر الجيوسياسية.",
        kind="write",
    )
    if regimes.empty:
        st.warning("Factor comparisons are unavailable for the selected date range.")
        return

    st.subheader("Portfolio Performance Under Different Market Conditions")
    explanation(
        "Each cell compares the average next-month return in two conditions. "
        "**pp = percentage points**, not percentage changes. For example, "
        "3.0% minus 1.5% equals +1.5 pp.",
        "كل خلية تعرض الفرق بين متوسط العائد الشهري في حالتين من ظروف السوق. "
        "اختصار pp يعني «نقاط مئوية». مثلًا 3% ناقص 1.5% يساوي 1.5 نقطة مئوية.",
    )
    comparisons = [
        ("Technical", "Technical trend (SMA20 / SMA50)", "Uptrend", "Downtrend"),
        ("Financial", "Financial volatility (VIX)", "High", "Low"),
        ("Real economy", "Real economy expansion / slowdown", "Expansion", "Slowdown"),
        ("Political", "Political geopolitical risk", "High", "Low"),
    ]
    overview = pd.DataFrame(index=METHODS)
    for label, factor_name, state_one, state_two in comparisons:
        section = regimes.loc[regimes["Factor"] == factor_name]
        by_state = section.pivot(
            index="Strategy", columns="Condition", values="Mean Monthly Return"
        ).reindex(METHODS)
        if state_one in by_state.columns and state_two in by_state.columns:
            overview[label] = 100 * (by_state[state_one] - by_state[state_two])
        else:
            overview[label] = float("nan")
    st.dataframe(overview.style.format("{:+.2f} pp", na_rep="N/A"), use_container_width=True)
    st.bar_chart(overview.T, use_container_width=True)
    explanation(
        "Difference in average NEXT-month return (percentage points, pp): "
        "Technical = uptrend − downtrend; Financial = high VIX − low VIX; "
        "Real economy = expansion − slowdown; Political = high GPR − low GPR. "
        "Positive means a higher historical return in the first condition; "
        "negative means a lower return. These differences do not show causation.",
        "الفرق في متوسط العائد للشهر التالي (بالنقاط المئوية): "
        "الفني = الاتجاه الصاعد ناقص الهابط؛ المالي = تقلبات VIX المرتفعة ناقص المنخفضة؛ "
        "الاقتصادي = التوسع ناقص التباطؤ؛ السياسي = مخاطر GPR المرتفعة ناقص المنخفضة. "
        "الإشارة الموجبة تعني عائدًا تاريخيًا أعلى في الحالة الأولى، والسالبة أقل. "
        "هذه مقارنة وصفية ولا تثبت السببية.",
    )

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
            explanation(factors[factor], FACTOR_ARABIC.get(factor, "شرح المؤشر."))
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
            explanation(
                "Each bar shows the average NEXT-month return in percent (%). "
                "The values are not cumulative gains.",
                "يمثّل كل عمود متوسط عائد الشهر التالي كنسبة مئوية، وليس العائد التراكمي.",
            )

            st.markdown("#### Monthly risk (volatility)")
            risks_chart = subset.pivot(
                index="Strategy",
                columns="Condition",
                values="Monthly Volatility",
            ).reindex(METHODS) * 100
            st.bar_chart(risks_chart, use_container_width=True)
            explanation(
                "Monthly volatility in percent (%). Lower indicates less historical variation.",
                "التقلب الشهري معروض بالنسبة المئوية. كلما انخفضت القيمة، كانت تذبذبات العوائد التاريخية أقل.",
            )

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
            explanation(
                "Months = number of realized monthly returns in that condition.",
                "Months تعني عدد الأشهر التي ظهرت فيها تلك الحالة؛ حجم العينة مهم لتفسير النتائج.",
            )

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
                    explanation(
                        f"When **{focus}** was observed, **{best['Strategy']}** had "
                        f"the highest monthly Sharpe ratio in this backtest "
                        f"({best['Monthly Sharpe']:.2f}; {int(best['Months'])} months). "
                        "Compare the return and volatility bars before interpreting that ranking.",
                        f"عند ظهور حالة {focus}، حققت محفظة {best['Strategy']} أعلى نسبة شارب شهرية "
                        f"ضمن الاختبار التاريخي ({best['Monthly Sharpe']:.2f}) بناءً على "
                        f"{int(best['Months'])} شهرًا. قارني العائد بالتقلب قبل اختيار طريقة.",
                        kind="info",
                    )
                else:
                    explanation(
                        "This condition has limited observations or missing risk estimates. "
                        "A method ranking is not reliable enough to highlight.",
                        "عدد الأشهر قليل أو بعض تقديرات المخاطر غير متوفرة؛ لذلك لا يمكن إبراز ترتيب موثوق للطرق.",
                        kind="info",
                    )

            explanation(
                "These are historical associations, NOT proof that any factor caused a "
                "portfolio gain or loss. High/low thresholds use training-period data only. "
                "Every method is compared over the same dates.",
                "توضح النتائج علاقات تاريخية ولا تثبت أن العامل سبّب ربحًا أو خسارة. "
                "حدود الحالات المرتفعة والمنخفضة محسوبة من بيانات التدريب، "
                "وجميع المحافظ تُقارن في الفترة نفسها.",
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
    explanation(
        "**Recommendation:** Use factor comparisons to assess whether a strategy's "
        "historical results were consistent across different conditions, not to predict "
        "which method will win next month.",
        "التوصية: قارني مدى ثبات نتائج كل محفظة بين ظروف السوق المختلفة، "
        "ولا تستخدمي هذا الجدول للتنبؤ بالطريقة التي ستتفوق الشهر المقبل.",
        kind="markdown",
    )
