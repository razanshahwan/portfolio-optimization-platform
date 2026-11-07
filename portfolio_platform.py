"""Friendly portfolio builder and reproducible Chapter 4 research dashboard.
Deployed at the same Streamlit app URL. Underlying calculations: thesis_engine.py.
"""
import io
import zipfile

import pandas as pd
import streamlit as st
import yfinance as yf
from market_factor_view import render_factor_dashboard
from ui_translation import explanation, arabic_below

from thesis_engine import (
    DEFAULT, GROUPS, get_prices, technical_panel,
    fundamental_economic_factors, geopolitical_factor, full_panel,
    evaluate_portfolios, current_allocations,
    metrics, prediction_errors, group_importance, regime_comparison,
    last_complete_month,
)

st.set_page_config(
    page_title="Portfolio Optimization Platform",
    page_icon="📊",
    layout="wide",
)

KNOWN = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEFA": "iShares Core MSCI EAFE ETF",
    "VNQ": "Vanguard Real Estate ETF",
    "USO": "United States Oil Fund",
    "KO": "The Coca-Cola Company",
}


@st.cache_data(ttl=86400, show_spinner=False)
def full_name(symbol):
    if symbol in KNOWN:
        return KNOWN[symbol]
    try:
        info = yf.Ticker(symbol).get_info()
        return info.get("longName") or info.get("shortName") or "Name unavailable — check symbol"
    except Exception:
        return "Name unavailable — check symbol"


@st.cache_data(ttl=3600, show_spinner=False)
def market_data(symbols, start):
    return get_prices(list(symbols), start)


@st.cache_data(ttl=21600, show_spinner=False)
def macro_data():
    return fundamental_economic_factors(), geopolitical_factor()


@st.cache_data(ttl=3600, show_spinner=False)
def research_run(monthly, daily, fred, gpr):
    panel=full_panel(technical_panel(daily,monthly),fred,gpr)
    results=evaluate_portfolios(monthly,panel)
    current=current_allocations(monthly,results["panel"],results["parameters"])
    return results,current


def to_zip(tables):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,table in tables.items():
            if table is not None and isinstance(table,pd.DataFrame):
                z.writestr(name+".csv",table.to_csv(index=True).encode("utf-8-sig"))
        z.writestr("README.txt",(
            "Portfolio Optimization Platform – Historical results and data\n"
            "All performance risk measures are monthly; cumulative returns cover test only.\n"
            "Prices: Yahoo Finance adjusted close; macro: FRED; GPR: Caldara-Iacoviello.\n"
            "Market prices are updated on rerun; current incomplete month excluded.\n"
            "FRED economic releases are lagged 2 months to reduce release-date leakage.\n"
            "Historical revised FRED values are NOT point-in-time vintage data.\n"
            "Backtest excludes taxes, fees and bid-ask spreads; no trading guarantee.\n"
        ).encode("utf-8"))
    return buffer.getvalue()


st.image("assets/portfolio_hero.svg", use_container_width=True)
st.title("Portfolio Optimization Platform")
arabic_below("منصة تحسين المحافظ الاستثمارية: مقارنة الطرق الثلاث وتحليل المخاطر وظروف السوق.")
explanation(
    "Build portfolios, compare three methods, and see how market conditions relate to performance.",
    "ابنِ محفظتك الاستثمارية، وقارن بين ثلاث طرق، وتعرّف إلى أداء المحافظ تحت ظروف السوق المختلفة.",
)
explanation(
    "**1. Pick assets → 2. Choose a goal → 3. Compare portfolio methods.**",
    "١. اختر الأسهم والصناديق → ٢. حدّد هدف المقارنة → ٣. قارن أداء المحافظ.",
    kind="markdown",
)
explanation(
    "Enter the symbols you want to invest in. The platform shows the complete "
    "company or fund names and compares **Mean–Variance**, **Random Forest**, "
    "and **XGBoost** using a historical monthly backtest.",
    "أدخل رموز الأسهم أو صناديق ETF التي تختارها. ستظهر الأسماء الكاملة للأصول "
    "ومقارنة تاريخية بين Mean–Variance وRandom Forest وXGBoost.",
    kind="write",
)

st.toggle(
    "🌐 Translate explanations to Arabic / إظهار الترجمة العربية",
    value=False, key="show_arabic",
    help="Show Arabic text under the English explanations throughout the website.",
)

with st.sidebar:
    st.header("Step 1 · Your assets")
    if "symbol_text" not in st.session_state:
        st.session_state["symbol_text"]=", ".join(DEFAULT)
    if st.button("Restore the 6 default ETFs",use_container_width=True):
        st.session_state["symbol_text"]=", ".join(DEFAULT)
    typed=st.text_area(
        "Stock / ETF symbols",key="symbol_text",height=92,
        help="Type ticker symbols separated by commas or new lines. Example: KO, SPY, GLD",
    )
    tickers=list(dict.fromkeys(
        part.strip().upper()
        for part in typed.replace("\n",",").split(",") if part.strip()
    ))
    if len(tickers)>25:
        st.warning("For performance, select up to 25 different assets at a time.")
    if tickers:
        explanation(
            "Full names appear after pressing Enter or clicking outside the input.",
            "سيظهر الاسم الكامل للسهم أو الصندوق بعد الضغط على Enter أو النقر خارج خانة الإدخال.",
        )
        st.dataframe(
            pd.DataFrame([{"Symbol":x,"Full name":full_name(x)} for x in tickers]),
            hide_index=True,use_container_width=True,
        )
    st.header("Step 2 · Your goal")
    goal=st.selectbox(
        "What would you prefer to compare?",
        ["Balance return and risk","Higher historical return","Lower historical risk"],
        help="This only highlights a backtest winner. It does not predict future performance.",
    )
    st.header("Step 3 · Run")
    start=st.date_input("Start date",value=pd.Timestamp("2015-01-01"),
                        max_value=last_complete_month().date())
    investment=st.number_input("Example investment (USD)",
                               min_value=100.0,value=10000.0,step=500.0)
    explanation(
        "Returns are calculated through the latest *completed* calendar month.",
        "تُحسب العوائد حتى آخر شهر تقويمي مكتمل، دون احتساب الشهر الجاري غير المنتهي.",
    )
    with st.expander("What do these methods mean?"):
        explanation(
            "**Mean–Variance:** classic allocation from average historical returns and risk.",
            "المتوسط–التباين: يوزّع الاستثمار بناءً على متوسط العوائد التاريخية والمخاطر.",
            kind="write",
        )
        explanation(
            "**Random Forest:** a machine-learning model predicting next-month returns.",
            "الغابة العشوائية: نموذج تعلم آلي يتنبأ بعوائد الشهر التالي.",
            kind="write",
        )
        explanation(
            "**XGBoost:** a second machine-learning model for return prediction.",
            "إكس جي بوست: نموذج تعلم آلي آخر للتنبؤ بالعوائد.",
            kind="write",
        )
        explanation(
            "The goal selector highlights historical results only; it does not personalize forecasts.",
            "اختيار الهدف يحدد النتيجة التاريخية التي نُبرزها، وليس توصية استثمارية شخصية أو ضمانًا للعوائد.",
            kind="write",
        )
    run=st.button("Build & compare portfolios",type="primary",
                  use_container_width=True,disabled=len(tickers)<2 or len(tickers)>25)

if run:
    st.session_state["research_settings"]={
        "symbols":tuple(tickers),"start":str(start),"investment":float(investment)
    }

st.subheader("Selected stocks and ETFs")
st.dataframe(pd.DataFrame([
    {"Symbol":t,"Full name":full_name(t),"Category":"Fund or company"}
    for t in tickers
]),hide_index=True,use_container_width=True)

if "research_settings" not in st.session_state:
    explanation(
        "The default list includes SPY, GLD, TLT, IEFA, VNQ and USO. "
        "Change any symbol, select your goal, then click **Build & compare portfolios**.",
        "القائمة الافتراضية تضم ستة صناديق استثمارية. يمكنك تغيير أي رمز وتحديد هدفك ثم تشغيل المقارنة.",
        kind="info",
    )
    st.stop()

settings=st.session_state["research_settings"]
symbols=settings["symbols"]
amount=settings["investment"]
if tuple(tickers)!=symbols or str(start)!=settings["start"] or float(investment)!=amount:
    st.warning("You have changed the inputs. Click **Build & compare portfolios** to apply the new settings.")

try:
    with st.spinner("Downloading updated adjusted market prices..."):
        daily, monthly=market_data(symbols,settings["start"])
    if set(monthly.columns)!=set(symbols):
        unavailable=[x for x in symbols if x not in monthly.columns]
        st.warning("These symbols were excluded because they lacked sufficient historical data: "
                   +", ".join(unavailable))
    with st.spinner("Retrieving FRED variables and the official Geopolitical Risk Index..."):
        fred,gpr=macro_data()
    with st.spinner("Training Random Forest / XGBoost and testing monthly portfolio decisions..."):
        result,forward=research_run(monthly,daily,fred,gpr)
except Exception as exc:
    st.error(f"Could not finish this analysis: {exc}")
    st.info("The complete comparison requires Yahoo Finance, FRED and official GPR data. "
            "A source outage is displayed rather than silently replacing economic factors.")
    st.stop()

backtest=result["returns"]
performance=metrics(result,monthly)
forecast_errors=prediction_errors(result["predictions"])
importance=group_importance(result["importance"])
regimes=regime_comparison(result)
latest_panel=result["panel"]
period=result["dates"]
st.caption(
    f"Market data through: **{monthly.index.max():%B %Y}** · "
    f"Complete factor-backed panel through: **{latest_panel['Date'].max():%B %Y}** · "
    f"Out-of-sample realized returns: **{backtest.index.min():%b %Y}–{backtest.index.max():%b %Y}**"
)

eligible=[x for x in ("Mean-Variance","Random Forest","XGBoost") if x in performance.index]
if eligible:
    if goal=="Higher historical return":
        winner=performance.loc[eligible,"Cumulative Return"].idxmax()
        reason="highest cumulative return"
    elif goal=="Lower historical risk":
        winner=performance.loc[eligible,"Monthly Volatility"].idxmin()
        reason="lowest monthly volatility"
    else:
        winner=performance.loc[eligible,"Monthly Sharpe"].idxmax()
        reason="highest monthly Sharpe ratio"
    explanation(
        f"**Historical result for your goal:** {winner} had the {reason} "
        "during this test period. This is a comparison, not a personal investment recommendation.",
        f"وفقًا لهدفك المختار، حققت محفظة {winner} أفضل قيمة للمؤشر المستخدم خلال فترة الاختبار التاريخية. "
        "هذه مقارنة تعليمية وليست توصية استثمارية شخصية.",
        kind="info",
    )

a,b,c,d=st.columns(4)
a.metric("Assets",len(monthly.columns))
b.metric("Testing months",len(backtest))
c.metric("Portfolio methods",len(eligible))
d.metric("Last completed price month",monthly.index.max().strftime("%b %Y"))

tabs=st.tabs([
    "📈 Performance",
    "🧩 Allocations",
    "🌍 Four market factors",
    "🔎 More insights & downloads",
])
with tabs[0]:
    st.subheader("How did the portfolios grow?")
    wealth=(1+backtest).cumprod()
    st.line_chart(wealth)
    explanation(
        "Growth of $1 invested at the start of the common out-of-sample period.",
        "يعرض الرسم نمو قيمة دولار واحد خلال فترة الاختبار المشتركة بين المحافظ الثلاث.",
    )
    st.subheader(f"Illustrative growth of ${amount:,.0f}")
    st.line_chart(wealth*amount)
    st.subheader("Monthly portfolio returns")
    st.line_chart(backtest)
    explanation(
        "Returns are historical and exclude transaction costs and taxes.",
        "العوائد المعروضة تاريخية ولا تتضمن تكاليف التداول أو الضرائب.",
    )

with tabs[0]:
    st.subheader("Compare the three portfolio methods")
    explanation(
        "Mean–Variance uses historical average returns. Random Forest and XGBoost predict next-month ETF returns.",
        "تعتمد Mean–Variance على متوسط العوائد السابقة؛ بينما يتنبأ Random Forest وXGBoost بعائد الشهر التالي.",
    )
    st.subheader("Portfolio performance · Monthly risk")
    format_pct={x:"{:.2%}" for x in [
        "Cumulative Return","Mean Monthly Return","Monthly Volatility",
        "Historical VaR 95%","Maximum Drawdown"]}
    format_pct["Monthly Sharpe"]="{:.3f}"
    if "Beta" in performance:
        format_pct["Beta"]="{:.3f}"
    st.dataframe(performance.style.format(format_pct),
                 use_container_width=True)
    st.subheader("Cumulative return by method")
    st.bar_chart(performance["Cumulative Return"])
    st.subheader("Monthly volatility by method")
    st.bar_chart(performance["Monthly Volatility"])
    explanation(
        "All methods use the same monthly covariance estimator (Ledoit–Wolf, 36 prior months), "
        "the same long-only/full-investment constraints, and the same testing dates. "
        "No maximum weight per individual asset is imposed.",
        "تستخدم المحافظ الثلاث طريقة Ledoit–Wolf لتقدير المخاطر من عوائد آخر 36 شهرًا، "
        "وتخضع للقيود وفترة الاختبار نفسها. لا يوجد سقف أعلى محدد لوزن أصل واحد.",
    )
    explanation(
        "**Recommendation:** Compare risk-adjusted performance and monthly volatility "
        "alongside cumulative returns. The historically strongest method is not guaranteed "
        "to perform best in the future.",
        "التوصية: لا تكتفِ بالعائد التراكمي؛ قارِن أيضًا شارب والتقلبات الشهرية. "
        "الأفضل تاريخيًا ليس بالضرورة الأفضل مستقبلًا.",
        kind="markdown",
    )

with tabs[1]:
    if forward.empty:
        st.warning("Latest allocations are not available for this dataset.")
    else:
        latest_date=forward["FormationDate"].max()
        st.subheader(f"Estimated portfolio allocations at {latest_date:%B %Y}")
        explanation(
            "Model-estimated allocations for the next month; not realized returns or a trade recommendation.",
            "أوزان تقديرية للشهر التالي وليست أرباحًا محققة أو توصية مباشرة بالشراء والبيع.",
        )
        w=forward.pivot(index="Ticker",columns="Strategy",values="Weight").fillna(0)
        st.bar_chart(w)
        st.dataframe(w.style.format("{:.1%}"),use_container_width=True)
        selected=st.selectbox("Explore a portfolio method",list(w.columns))
        detail=pd.DataFrame({
            "Symbol":w.index,
            "Full name":[full_name(x) for x in w.index],
            "Weight":w[selected].values,
            "Illustrative USD":w[selected].values*amount,
        })
        st.dataframe(detail.style.format({"Weight":"{:.1%}","Illustrative USD":"${:,.2f}"}),
                     hide_index=True,use_container_width=True)
        explanation(
            "Some methods may concentrate a large percentage in one asset because weights are not capped.",
            "قد تتركز نسبة كبيرة من المحفظة في أصل واحد لأنه لا يوجد حد أعلى لوزنه.",
        )
    st.subheader("Monthly changes in portfolio allocations")
    history_w=result["weights"].copy()
    chosen=st.selectbox("History for",["Mean-Variance","Random Forest","XGBoost"])
    historic=history_w[history_w["Strategy"]==chosen].pivot(
        index="Date",columns="Ticker",values="Weight")
    st.area_chart(historic)
    explanation(
        "**Recommendation:** Check for large single-asset weights and how allocations change over time before interpreting any method as diversified.",
        "التوصية: راجع وزن كل سهم أو صندوق وتغير توزيعه عبر الزمن قبل اعتبار المحفظة متنوعة بما يكفي.",
        kind="markdown",
    )

with tabs[2]:
    render_factor_dashboard(regimes)

with tabs[3]:
    st.subheader("Understanding the return forecasts")
    st.subheader("Monthly return prediction errors")
    st.dataframe(forecast_errors.style.format({"MAE":"{:.4f}","RMSE":"{:.4f}"}),
                 use_container_width=True,hide_index=True)
    explanation(
        "Both MAE and RMSE are in decimal monthly return units. Lower is better.",
        "مقياسا MAE وRMSE يوضحان خطأ التنبؤ بالعائد الشهري، وكلما انخفضت قيمتهما كانت الدقة أفضل.",
    )
    selected_model=st.selectbox("Inspect predicted vs actual monthly returns",
                                ["Random Forest","XGBoost"])
    forecast=result["predictions"]
    sample=forecast[forecast["Model"]==selected_model].copy()
    st.scatter_chart(sample,x="Actual",y="Predicted",color="Ticker")
    st.subheader("Average model feature importance")
    selected_importance=importance[importance["Model"]==selected_model]
    top_features=selected_importance.sort_values("Importance",ascending=False).head(18)
    st.bar_chart(top_features.set_index("Feature")["Importance"])
    st.subheader("Which inputs contribute to return predictions?")
    group_scores=selected_importance.groupby("Group")["Importance"].sum().sort_values(ascending=False)
    st.bar_chart(group_scores)
    explanation(
        "Tree-model importance is predictive association, not evidence of causal effect.",
        "أهمية المتغيرات تعبّر عن دورها في التنبؤ، ولا تثبت علاقة سببية.",
    )
    st.subheader("Training-only time-series cross-validation")
    st.dataframe(result["cv"],use_container_width=True,hide_index=True)

with tabs[3]:
    st.divider()
    st.subheader("Detailed factor and market conditions")
    st.subheader("Out-of-sample performance by financial and political conditions")
    st.dataframe(regimes.style.format({
        "Mean Monthly Return":"{:.2%}","Monthly Volatility":"{:.2%}",
        "Monthly Sharpe":"{:.3f}"}),hide_index=True,use_container_width=True)
    for feature in ("VIX","GPR"):
        st.subheader(feature+" at portfolio-formation time")
        factor=latest_panel.drop_duplicates("Date").set_index("Date")[feature]
        st.line_chart(factor)
    explanation(
        "High/low thresholds use TRAINING-period medians only. Each realized next-month "
        "portfolio return is associated with the factors known at formation time.",
        "تُحدد الحدود الفاصلة بين الظروف المرتفعة والمنخفضة من فترة التدريب فقط، "
        "ثم يُقارن عائد الشهر التالي مع الظروف المعروفة عند تكوين المحفظة.",
    )
    explanation(
        "FRED inflation, industrial growth and unemployment are conservatively lagged "
        "two months. FRED's revised historical series are not point-in-time vintages; "
        "an ALFRED vintage robustness check remains necessary for strict academic validation.",
        "تُؤخّر بيانات التضخم والنمو الصناعي والبطالة شهرين بصورة احترازية، "
        "لكن بعض بيانات FRED التاريخية منقحة بعد نشرها، لذلك يلزم التحقق من النسخ الأصلية للبيانات.",
    )

with tabs[3]:
    st.divider()
    st.subheader("Download your results")
    st.subheader("Research data and methodology")
    st.write(
        "Ticker universe: "+", ".join(monthly.columns)
        +". Daily adjusted closes from Yahoo Finance; FRED macro and market variables; "
        "official Caldara–Iacoviello monthly GPR. A pooled panel feeds "
        "Random Forest and XGBoost regression models. The earliest 70% of complete "
        "formation months are assigned to the training period; the final 30% to testing."
    )
    table=pd.DataFrame([{"Stage":key,"Date":str(pd.Timestamp(val).date())}
                        for key,val in period.items()])
    st.dataframe(table,hide_index=True,use_container_width=True)
    st.subheader("Panel data preview")
    st.dataframe(latest_panel.tail(18),use_container_width=True,hide_index=True)
    st.subheader("Download numerical tables and chart data")
    tables={
        "01_monthly_prices":monthly,
        "02_monthly_returns":monthly.pct_change(fill_method=None),
        "03_research_panel":latest_panel,
        "04_out_of_sample_returns":backtest,
        "05_performance_metrics":performance,
        "06_prediction_accuracy":forecast_errors,
        "07_predictions":result["predictions"],
        "08_historical_weights":result["weights"],
        "09_latest_weights":forward,
        "10_factor_importance":importance,
        "11_factor_regimes":regimes,
        "12_training_cv":result["cv"],
    }
    st.download_button("Download all research tables (ZIP / CSV)",
                       data=to_zip(tables),file_name="portfolio_analysis_results.zip",
                       mime="application/zip",use_container_width=True)
    explanation(
        "The ZIP includes the data used to create the comparisons and charts. "
        "Check source coverage before interpreting any results.",
        "يتضمن ملف ZIP الجداول التي بُنيت عليها المقارنات والرسومات؛ يُفضّل التأكد من اكتمال البيانات قبل الاستنتاج.",
    )
    explanation(
        "**Recommendation:** Use the downloadable tables to verify how often each "
        "condition occurred and compare both forecast errors and portfolio risk.",
        "التوصية: استخدم الملفات للتحقق من عدد الأشهر في كل حالة، وقارن أخطاء التنبؤ ومخاطر المحافظ.",
        kind="markdown",
    )

st.divider()
explanation(
    "Educational and research tool, not investment advice. Historical results can differ "
    "materially from future performance. Source errors and incomplete factor observations "
    "are shown explicitly instead of fabricating missing data.",
    "هذه أداة تعليمية وتحليلية وليست نصيحة استثمارية. قد تختلف النتائج المستقبلية عن التاريخية، "
    "ويتم توضيح أخطاء المصادر أو نقص بيانات العوامل بدل تقديم بيانات غير حقيقية.",
)
