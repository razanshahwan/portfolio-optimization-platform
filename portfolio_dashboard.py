
import streamlit as st
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import investpy
except ImportError:
    investpy = None

st.title("Dynamic Portfolio Optimization Dashboard")

st.write("This dashboard will compare portfolio strategies:")
st.write("- Equal Weight")
st.write("- Mean-Variance")
st.write("- Random Forest")
st.write("- XGBoost")

st.write("---")

source = st.selectbox("اختر مصدر بيانات الأسهم:", ["Yahoo Finance", "Investing.com"])

ticker = st.text_input(
    "رمز السهم أو اسم السهم:",
    value="AAPL" if source == "Yahoo Finance" else "Apple"
)

country = None
if source == "Investing.com":
    country = st.text_input("الدولة (Investing.com):", value="united states")

period = st.selectbox(
    "اختر فترة البيانات:",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
)

st.write("---")

def get_date_range(period):
    end = datetime.today()
    if period == "1mo":
        start = end - timedelta(days=30)
    elif period == "3mo":
        start = end - timedelta(days=90)
    elif period == "6mo":
        start = end - timedelta(days=180)
    elif period == "1y":
        start = end - timedelta(days=365)
    elif period == "2y":
        start = end - timedelta(days=730)
    elif period == "5y":
        start = end - timedelta(days=1825)
    else:
        start = end - timedelta(days=3650)
    return start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y")

if st.button("تحميل بيانات السهم"):
    if source == "Yahoo Finance":
        if yf is None:
            st.error("يجب تثبيت مكتبة yfinance أولاً لتجلب بيانات Yahoo Finance. نفذ: pip install yfinance")
        else:
            try:
                df = yf.Ticker(ticker).history(period=period)
                if df.empty:
                    st.warning("لم يتم العثور على بيانات للسهم المدخل من Yahoo Finance.")
                else:
                    st.success("تم تحميل البيانات من Yahoo Finance")
                    st.line_chart(df["Close"])
                    st.dataframe(df.reset_index())
            except Exception as err:
                st.error(f"حدث خطأ أثناء جلب البيانات من Yahoo Finance: {err}")
    else:
        if investpy is None:
            st.error("يجب تثبيت مكتبة investpy أولاً لتجلب بيانات Investing.com. نفذ: pip install investpy")
        elif not country:
            st.warning("يرجى إدخال الدولة المرتبطة بالأسهم على Investing.com.")
        else:
            try:
                start_date, end_date = get_date_range(period)
                df = investpy.stocks.get_stock_historical_data(
                    stock=ticker,
                    country=country,
                    from_date=start_date,
                    to_date=end_date,
                )
                if df.empty:
                    st.warning("لم يتم العثور على بيانات للسهم المدخل من Investing.com.")
                else:
                    st.success("تم تحميل البيانات من Investing.com")
                    st.line_chart(df["Close"])
                    st.dataframe(df.reset_index())
            except Exception as err:
                st.error(
                    "حدث خطأ أثناء جلب البيانات من Investing.com. تأكد أن اسم السهم والدولة صحيحان، وأن مكتبة investpy محدثة."
                )
                st.exception(err)

st.write("---")
st.write(
    "يمكنك الآن اختيار مصدر البيانات بين Yahoo Finance و Investing.com. إذا اخترت Investing.com، أدخل اسم السهم والدولة كما يظهر في موقع Investing.com."
)
