"""Educational portfolio allocation signals for the public website only.

This module does NOT generate trades or alter the research backtest, model
training, downloadable research tables, or any thesis methodology.
"""
import numpy as np
import pandas as pd
import streamlit as st

from ui_translation import explanation

METHODS = ["Mean-Variance", "Random Forest", "XGBoost"]


def _recent_market_conditions(panel, train_end):
    """Describe formation-month inputs without claiming factor causation."""
    if panel.empty:
        return pd.DataFrame(columns=["Factor group", "Latest condition", "Interpretation"])
    p = panel.copy()
    dates = pd.to_datetime(p["Date"])
    latest = p.loc[dates == dates.max()]
    baseline = p.loc[dates <= pd.Timestamp(train_end)]
    if latest.empty or baseline.empty:
        return pd.DataFrame(columns=["Factor group", "Latest condition", "Interpretation"])
    b = baseline.groupby("Date")[["VIX", "GPR"]].mean()
    current_vix = float(latest["VIX"].mean())
    current_gpr = float(latest["GPR"].mean())
    trend = "Uptrend" if float(latest["SMA20_SMA50"].mean()) > 1 else "Downtrend"
    industry = float(latest["IndustrialGrowth_YoY"].mean())
    vix_state = "Higher" if current_vix > b["VIX"].median() else "Lower"
    gpr_state = "Higher" if current_gpr > b["GPR"].median() else "Lower"
    return pd.DataFrame([
        {"Factor group": "Technical", "Latest condition": trend,
         "Interpretation": "Average selected-asset moving-average trend"},
        {"Factor group": "Financial", "Latest condition": f"{vix_state} VIX",
         "Interpretation": "Compared with the training-period median"},
        {"Factor group": "Real economy",
         "Latest condition": f"Industrial growth {industry:+.1f}% YoY",
         "Interpretation": "Most recently available lagged macro reading"},
        {"Factor group": "Political", "Latest condition": f"{gpr_state} GPR",
         "Interpretation": "Compared with the training-period median"},
    ])


def render_allocation_signals(forward, panel, dates, amount, full_name):
    """Illustrative weight gap vs user-entered holdings, never broker orders."""
    st.header("Investment Signals & Rebalancing")
    explanation(
        "Compare the model's **target portfolio weights** with your current holdings. "
        "The result is an educational allocation signal—not an instruction to trade.",
        "قارن الأوزان المقترحة من النموذج مع توزيع محفظتك الحالي. "
        "النتائج إشارات تعليمية لإعادة التوازن وليست أوامر شراء أو بيع.",
        kind="write",
    )
    if forward is None or forward.empty:
        st.warning("Target portfolio allocations are unavailable for this dataset.")
        return

    methods = [m for m in METHODS if m in forward["Strategy"].unique()]
    if not methods:
        st.warning("No portfolio method is available.")
        return

    strategy = st.selectbox(
        "Choose the method to explore",
        methods,
        index=0,
        key="allocation_signals_method",
    )
    subset = forward.loc[forward["Strategy"] == strategy].copy()
    subset = subset.drop_duplicates("Ticker", keep="last").sort_values("Ticker")
    target = pd.Series(
        pd.to_numeric(subset["Weight"], errors="coerce").to_numpy(),
        index=subset["Ticker"],
    )
    if target.isna().any() or (target < 0).any() or target.sum() <= 0:
        st.error("The selected method did not produce valid target allocations.")
        return

    target = (target / target.sum()).sort_values(ascending=False)
    formation_date = pd.to_datetime(subset["FormationDate"].iloc[0])
    explanation(
        f"These are **{strategy}** target weights estimated using information available "
        f"through **{formation_date:%B %Y}**. They are not realized future returns.",
        f"هذه أوزان محفظة {strategy} محسوبة باستخدام البيانات المتاحة حتى "
        f"{formation_date:%Y-%m}. وهي ليست عوائد مستقبلية محققة.",
    )
    table = pd.DataFrame({
        "Symbol": target.index,
        "Full name": [full_name(t) for t in target.index],
        "Target weight (%)": np.round(target.values * 100, 2),
        "Illustrative amount ($)": np.round(target.values * amount, 2),
    })
    st.subheader("Model target allocations")
    st.dataframe(
        table.style.format({
            "Target weight (%)": "{:.2f}%",
            "Illustrative amount ($)": "${:,.2f}",
        }), hide_index=True, use_container_width=True,
    )
    st.bar_chart(target.rename("Target weight"), use_container_width=True)
    explanation(
        "The chart shows how the selected method distributes the portfolio, "
        "not a forecast of each asset's price. Large weights can mean concentration risk.",
        "يوضح الرسم توزيع المحفظة بين الأصول، وليس توقعًا لسعر كل أصل. "
        "الأوزان الكبيرة قد تعني مخاطر تركز أعلى.",
    )

    st.subheader("Compare with a portfolio you already hold")
    mode = st.radio(
        "Do you want to enter current holdings?",
        ["Just exploring", "I have a portfolio"],
        horizontal=True,
        key="allocation_signals_mode",
    )
    if mode == "I have a portfolio":
        explanation(
            "Enter the current percentage weight of **each selected asset**. "
            "The total must equal 100%. Do not enter money amounts or share counts.",
            "أدخل نسبة وزن كل أصل في محفظتك الحالية، بحيث يكون مجموع الأوزان 100%. "
            "لا تُدخل مبالغ مالية أو عدد الأسهم.",
        )
        current = pd.DataFrame({
            "Symbol": list(target.index),
            "Full name": [full_name(t) for t in target.index],
            "Current weight (%)": [0.0] * len(target),
        })
        edited = st.data_editor(
            current, hide_index=True, use_container_width=True,
            disabled=["Symbol", "Full name"], num_rows="fixed",
            column_config={
                "Current weight (%)": st.column_config.NumberColumn(
                    "Current weight (%)", min_value=0.0, max_value=100.0,
                    step=0.5, format="%.1f",
                ),
            },
            key="allocation_current_" + "_".join(target.index),
        )
        current_w = pd.to_numeric(
            edited["Current weight (%)"], errors="coerce",
        )
        valid = (
            current_w.notna().all()
            and current_w.between(0, 100).all()
            and abs(float(current_w.sum()) - 100) <= 0.25
        )
        if not valid:
            st.info(
                f"Enter your actual weights so they add up to 100%. "
                f"Current total: {float(current_w.sum()):.1f}%."
            )
        else:
            st.subheader("Allocation changes to review")
            threshold = st.slider(
                "Hold range (percentage points)", min_value=0.5,
                max_value=5.0, value=2.0, step=0.5,
                help="Differences within this range display Hold for simplicity.",
            )
            target_pct = target.values * 100
            difference = target_pct - current_w.to_numpy()
            signal = np.where(
                difference > threshold, "Increase",
                np.where(difference < -threshold, "Reduce", "Hold")
            )
            signals = pd.DataFrame({
                "Symbol": target.index,
                "Full name": [full_name(t) for t in target.index],
                "Current (%)": np.round(current_w.to_numpy(), 2),
                "Target (%)": np.round(target_pct, 2),
                "Gap (pp)": np.round(difference, 2),
                "Signal": signal,
            })
            st.dataframe(
                signals.style.format({
                    "Current (%)": "{:.2f}%",
                    "Target (%)": "{:.2f}%",
                    "Gap (pp)": "{:+.2f}",
                }), use_container_width=True, hide_index=True,
            )
            comparison = pd.DataFrame({
                "Current": current_w.to_numpy(),
                "Target": target_pct,
            }, index=target.index)
            st.bar_chart(comparison, use_container_width=True)
            explanation(
                "**Increase / Reduce / Hold** describe the difference between user-entered "
                "weights and model targets. They do not indicate that an asset's price will "
                "rise or fall. The hold range is a display rule, not a tested trading rule.",
                "تشير زيادة / تخفيض / احتفاظ إلى الفرق بين الوزن الحالي والوزن المقترح، "
                "ولا تعني أن سعر الأصل سيرتفع أو ينخفض. هامش الاحتفاظ قاعدة للعرض فقط "
                "وليس قاعدة تداول تم اختبارها.",
            )
            st.markdown("**Illustrative review:**")
            st.write(
                "A positive gap suggests the target weight is higher than the "
                "current allocation; a negative gap means it is lower. "
                "Consider concentration, transaction costs and your own circumstances "
                "before interpreting any allocation change."
            )
            if st.session_state.get("show_arabic", False):
                st.caption(
                    "الفرق الموجب يعني أن الوزن المقترح أعلى من الحالي، والسالب يعني أنه "
                    "أقل. يجب أخذ التركز وتكاليف التداول والظروف الشخصية في الاعتبار."
                )

    with st.expander("How do the four market-factor groups fit into the signals?"):
        explanation(
            "For Random Forest and XGBoost, technical, financial, real-economy, "
            "and political indicators contribute to expected-return estimates. "
            "Those estimates and historical portfolio risk determine target weights. "
            "Mean–Variance instead uses historical average returns. "
            "The factors are **not** independent Buy/Sell triggers.",
            "تستخدم Random Forest وXGBoost المؤشرات الفنية والمالية والاقتصادية "
            "والسياسية لتوقع العوائد، ثم تُستخدم التوقعات والمخاطر لحساب الأوزان. "
            "أما Mean–Variance فتعتمد على متوسط العوائد التاريخية. "
            "العوامل لا تُصدر أوامر شراء أو بيع منفردة.",
            kind="write",
        )
        context = _recent_market_conditions(panel, dates["TrainEnd"])
        if not context.empty:
            st.dataframe(context, hide_index=True, use_container_width=True)
            explanation(
                "Market conditions are snapshots at the latest eligible formation month. "
                "They are descriptive, not causal explanations for a particular weight.",
                "ظروف السوق تعرض أحدث البيانات المتاحة عند تكوين المحفظة، "
                "ولا تثبت أن عاملًا معينًا تسبب في إعطاء وزن محدد.",
            )

    explanation(
        "**Recommendation:** Treat these signals as a portfolio-comparison exercise. "
        "No real orders are placed; this feature is not part of the academic backtest "
        "and has not been separately validated as a trading strategy.",
        "التوصية: استخدم الإشارات لفهم إعادة توزيع الأوزان فقط؛ لا يجري تنفيذ "
        "أي عمليات تداول فعلية، ولم تُختبر هذه الإضافة بوصفها استراتيجية تداول مستقلة.",
        kind="markdown",
    )
