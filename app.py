import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📈 AI Stock Predictor", layout="wide", page_icon="🚀")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* 🌈 BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* 🏆 MAIN TITLE */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.subtitle {
    text-align: center;
    color: #a0aec0;
    font-size: 15px;
    margin-bottom: 30px;
}

/* 📦 METRIC CARDS */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 20px;
    padding: 24px 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-label {
    color: #a0aec0;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: #ffd200;
}

.metric-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

/* 📊 SECTION CARD */
.section-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 16px;
}

/* 🟢 SUCCESS BOX */
.predict-result {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 32px rgba(56, 239, 125, 0.25);
}

.predict-label {
    color: rgba(255,255,255,0.85);
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.predict-value {
    color: #ffffff;
    font-size: 52px;
    font-weight: 800;
}

/* 🔘 STREAMLIT OVERRIDES */
div[data-testid="stRadio"] label {
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.stButton > button {
    background: linear-gradient(90deg, #f7971e, #ffd200);
    color: #1a1a2e;
    font-weight: 800;
    font-size: 17px;
    border: none;
    border-radius: 14px;
    height: 54px;
    width: 100%;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(247, 151, 30, 0.4);
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 28px rgba(247, 151, 30, 0.6);
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 20px 0;
}

/* Radio + select text colors */
.stRadio > div > label > div > p {
    color: #e2e8f0 !important;
}

p, li, span, label {
    color: #cbd5e0;
}

h1, h2, h3 {
    color: #ffffff;
}

/* Number input background */
input[type=number] {
    background-color: #0f0c29 !important;
    color: #ffd200 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    background-color: #1a1a2e !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #ffd200 !important;
}

/* Success/error messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🚀 AI Stock Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict future stock prices using Machine Learning ✨</div>", unsafe_allow_html=True)


def get_last(df, col):
    """Safely get last numeric value from a column (handles MultiIndex too)."""
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(s.iloc[-1])


def flatten_columns(df):
    """
    yfinance sometimes returns MultiIndex columns like ('Close', 'AAPL').
    This function flattens them to simple names → 'Close', 'Open', etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📡 Select Data Source</div>", unsafe_allow_html=True)
data_source = st.radio("Where do you want to load data from?", ["📂 CSV File", "🌐 Live Market Data"], horizontal=True)
st.markdown("</div>", unsafe_allow_html=True)



if data_source == "📂 CSV File":
    try:
        df = pd.read_csv("data.csv")
        currency = "₹"
        st.success("✅ CSV file loaded successfully!")
    except FileNotFoundError:
        st.error("❌ 'data.csv' file not found! Please place it in the same folder.")
        st.stop()


else:
    stocks = {
        "🍎 Apple (AAPL)":     "AAPL",
        "🔍 Google (GOOGL)":   "GOOGL",
        "⚡ Tesla (TSLA)":     "TSLA",
        "📦 Amazon (AMZN)":    "AMZN",
        "💻 Microsoft (MSFT)": "MSFT",
        "🎮 NVIDIA (NVDA)":    "NVDA",
        "📱 Meta (META)":      "META",
    }

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏢 Select a Stock</div>", unsafe_allow_html=True)
    stock_name = st.selectbox("Choose a Company:", list(stocks.keys()))
    st.markdown("</div>", unsafe_allow_html=True)

    ticker = stocks[stock_name]

    with st.spinner(f"⏳ Fetching data for {stock_name}..."):
        raw_df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if raw_df is None or raw_df.empty:
        st.error("❌ No data found! Please check your internet connection or try again later.")
        st.stop()

    raw_df = flatten_columns(raw_df)
    df = raw_df.reset_index()
    currency = "$"
    st.success(f"✅ Live data loaded successfully for {stock_name}!")


df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Open", "High", "Low", "Close"])
df = df.reset_index(drop=True)

last_price = get_last(df, "Close")
last_open  = get_last(df, "Open")
last_high  = get_last(df, "High")
last_low   = get_last(df, "Low")


col1, col2, col3, col4 = st.columns(4)

cards = [
    ("💰", "Current Price",  f"{currency}{last_price:.2f}", col1),
    ("📈", "Today's High",   f"{currency}{last_high:.2f}",  col2),
    ("📉", "Today's Low",    f"{currency}{last_low:.2f}",   col3),
    ("📋", "Total Records",  str(len(df)),                  col4),
]

for icon, label, value, col in cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📋 Recent Stock Data</div>", unsafe_allow_html=True)

display_df = df[["Date", "Open", "High", "Low", "Close"]].tail(10).copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
for c in ["Open", "High", "Low", "Close"]:
    display_df[c] = display_df[c].apply(lambda x: f"{currency}{x:.2f}")

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📉 Closing Price Chart (1 Year)</div>", unsafe_allow_html=True)
chart_df = df.set_index("Date")[["Close"]].copy()
st.line_chart(chart_df, use_container_width=True, height=300)
st.caption("📌 This chart shows the closing price trend over the past 1 year.")
st.markdown("</div>", unsafe_allow_html=True)


model_df = df[["Open", "High", "Low", "Close"]].dropna()

for col in ["Open", "High", "Low", "Close"]:
    if isinstance(model_df[col], pd.DataFrame):
        model_df[col] = model_df[col].iloc[:, 0]
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

model_df = model_df.dropna()

X = model_df[["Open", "High", "Low"]].values
y = model_df["Close"].values

model = LinearRegression()
model.fit(X, y)

st.markdown("""
<div style='background: linear-gradient(90deg, #11998e, #38ef7d);
            border-radius: 14px; padding: 14px 20px; text-align:center;
            margin-bottom: 20px; font-weight:700; color:#1a1a2e; font-size:16px;'>
    🤖 AI Model Trained Successfully!
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>🔮 Predict Price</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#a0aec0; font-size:13px; margin-top:-10px;'>Enter values below and click Predict 👇</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    open_i = st.number_input(f"📂 Open Price ({currency})", value=round(last_open, 2), step=0.5)
with col2:
    high_i = st.number_input(f"⬆️ High Price ({currency})", value=round(last_high, 2), step=0.5)
with col3:
    low_i  = st.number_input(f"⬇️ Low Price ({currency})",  value=round(last_low,  2), step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Now!"):
    try:
        pred  = model.predict([[open_i, high_i, low_i]])
        value = float(pred[0])

        diff  = value - last_price
        arrow = "📈" if diff >= 0 else "📉"
        color_hex = "#38ef7d" if diff >= 0 else "#ff6b6b"
        direction = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"

        st.markdown(f"""
        <div class='predict-result'>
            <div class='predict-label'>🎯 Predicted Closing Price</div>
            <div class='predict-value'>{currency}{value:.2f}</div>
            <div style='font-size:16px; color:rgba(255,255,255,0.85); margin-top:8px;'>
                {arrow} Difference from current price: <span style='color:{color_hex}; font-weight:800;'>{currency}{direction}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f" Prediction error: {e}")

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center; padding: 30px 0 10px; color: rgba(255,255,255,0.2); font-size: 12px;'>
    Made with ❤️ using Streamlit & Scikit-learn &nbsp;|&nbsp; For Educational Use Only
</div>
""", unsafe_allow_html=True)


