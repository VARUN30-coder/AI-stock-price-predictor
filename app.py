import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }

/* ── Kill the white strip ── */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > * {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
#MainMenu { display: none !important; }
footer    { display: none !important; }

div[data-testid="stAppViewBlockContainer"] { padding-top: 2rem !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

/* ── Background ── */
.stApp { background: #f0f2f5; }

/* ── Title ── */
.main-title {
    font-size: 30px; font-weight: 700; color: #0d1117;
    letter-spacing: -0.5px; margin-bottom: 2px;
}
.subtitle { color: #6b7280; font-size: 13px; margin-bottom: 28px; }

/* ── Metric Cards ── */
.metric-card {
    background: #fff; border-radius: 12px; padding: 20px 18px;
    border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow .2s, transform .2s;
}
.metric-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.08); transform: translateY(-2px);
}
.metric-label {
    color: #9ca3af; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
}
.metric-value {
    font-size: 26px; font-weight: 700; color: #0d1117;
    font-family: 'DM Mono', monospace; letter-spacing: -0.5px;
}

/* ── Section Card ── */

.section-title {
    font-size: 14px; font-weight: 600; color: #0d1117;
    margin-bottom: 3px;
}
.section-desc { font-size: 12px; color: #9ca3af; margin-bottom: 16px; }

/* ── Trained Banner ── */
.trained-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 11px 16px; margin-bottom: 18px; font-size: 13px;
    font-weight: 600; color: #15803d;
}

/* ── Predict Result ── */
.predict-result {
    background: #0d1117; border-radius: 12px; padding: 30px 24px;
    text-align: center; margin-top: 18px; border: 1px solid #1f2937;
}
.predict-label {
    color: #6b7280; font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
}
.predict-value {
    color: #f9fafb; font-size: 44px; font-weight: 700;
    font-family: 'DM Mono', monospace; letter-spacing: -1px;
}
.predict-diff { font-size: 13px; margin-top: 8px; color: #6b7280; font-weight: 500; }

/* ── Streamlit overrides ── */
div[data-testid="stRadio"] label { color: #374151 !important; font-size: 14px !important; font-weight: 500 !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label { color: #374151 !important; font-weight: 500 !important; font-size: 13px !important; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #e5e7eb; }

.stButton > button {
    background: #0d1117; color: #f9fafb; font-weight: 600; font-size: 14px;
    border: none; border-radius: 10px; height: 46px; width: 100%;
    transition: all .2s; box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
.stButton > button:hover {
    background: #1f2937; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    transform: translateY(-1px);
}

input[type=number] {
    background-color: #f9fafb !important; color: #0d1117 !important;
    border-radius: 8px !important; border: 1px solid #d1d5db !important;
    font-weight: 600 !important; font-family: 'DM Mono', monospace !important;
}
div[data-testid="stSelectbox"] > div > div {
    background-color: #f9fafb !important; border: 1px solid #d1d5db !important;
    border-radius: 10px !important; color: #0d1117 !important;
}
div[data-testid="stAlert"] { border-radius: 10px; }
p, li, span { color: #374151; }
h1, h2, h3  { color: #0d1117; }

</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>AI Stock Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict future stock closing prices using linear regression on market data.</div>", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_last(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else 0.0

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Data Source ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Data Source</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Choose between a local CSV file or live market data via Yahoo Finance.</div>", unsafe_allow_html=True)
data_source = st.radio("", ["CSV File", "Live Market Data"], horizontal=True, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if data_source == "CSV File":
    try:
        df = pd.read_csv("data.csv")
        currency = "₹"
        st.success("CSV file loaded successfully.")
    except FileNotFoundError:
        st.error("'data.csv' not found. Please place it in the same directory.")
        st.stop()
else:
    us_stocks = {
        "Apple (AAPL)":     ("AAPL",   "$"),
        "Google (GOOGL)":   ("GOOGL",  "$"),
        "Tesla (TSLA)":     ("TSLA",   "$"),
        "Amazon (AMZN)":    ("AMZN",   "$"),
        "Microsoft (MSFT)": ("MSFT",   "$"),
        "NVIDIA (NVDA)":    ("NVDA",   "$"),
        "Meta (META)":      ("META",   "$"),
    }
    in_stocks = {
        "Reliance Industries (RELIANCE)":  ("RELIANCE.NS",  "₹"),
        "Tata Consultancy Services (TCS)": ("TCS.NS",       "₹"),
        "HDFC Bank (HDFCBANK)":            ("HDFCBANK.NS",  "₹"),
        "Infosys (INFY)":                  ("INFY.NS",      "₹"),
        "ICICI Bank (ICICIBANK)":          ("ICICIBANK.NS", "₹"),
        "Wipro (WIPRO)":                   ("WIPRO.NS",     "₹"),
        "Adani Enterprises (ADANIENT)":    ("ADANIENT.NS",  "₹"),
        "Bajaj Finance (BAJFINANCE)":      ("BAJFINANCE.NS","₹"),
        "HUL (HINDUNILVR)":                ("HINDUNILVR.NS","₹"),
        "State Bank of India (SBIN)":      ("SBIN.NS",      "₹"),
    }
    all_stocks = {**us_stocks, **in_stocks}

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Select Equity</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Live data fetched from Yahoo Finance — US stocks in $, Indian stocks (NSE) in ₹.</div>", unsafe_allow_html=True)
    stock_name = st.selectbox("", list(all_stocks.keys()), label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    ticker, currency = all_stocks[stock_name]

    with st.spinner(f"Fetching data for {stock_name}..."):
        raw_df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

    if raw_df is None or raw_df.empty:
        st.error("No data found. Check your internet connection and try again.")
        st.stop()

    raw_df = flatten_columns(raw_df)
    df = raw_df.reset_index()
    st.success(f"Live data loaded for {stock_name}.")


# ── Data Prep ────────────────────────────────────────────────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

last_price = get_last(df, "Close")
last_open  = get_last(df, "Open")
last_high  = get_last(df, "High")
last_low   = get_last(df, "Low")


# ── Metric Cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
cards = [
    ("Current Price", f"{currency}{last_price:.2f}", col1),
    ("52W High",      f"{currency}{last_high:.2f}",  col2),
    ("52W Low",       f"{currency}{last_low:.2f}",   col3),
    ("Trading Days",  str(len(df)),                  col4),
]
for label, value, col in cards:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)


# ── Recent Data Table ────────────────────────────────────────────────────────
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Recent Market Data</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Last 10 trading sessions.</div>", unsafe_allow_html=True)
display_df = df[["Date", "Open", "High", "Low", "Close"]].tail(10).copy()
display_df["Date"] = display_df["Date"].dt.strftime("%d %b %Y")
for c in ["Open", "High", "Low", "Close"]:
    display_df[c] = display_df[c].apply(lambda x: f"{currency}{x:.2f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)


# ── Professional Plotly Candlestick Chart ─────────────────────────────────────
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Price Chart — 1 Year</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Candlestick OHLC with MA20 and MA50 moving averages. Hover for details.</div>", unsafe_allow_html=True)

chart_df = df[["Date", "Open", "High", "Low", "Close"]].copy()
for c in ["Open", "High", "Low", "Close"]:
    chart_df[c] = pd.to_numeric(chart_df[c], errors="coerce")

chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
chart_df["MA50"] = chart_df["Close"].rolling(50).mean()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=chart_df["Date"],
    open=chart_df["Open"],
    high=chart_df["High"],
    low=chart_df["Low"],
    close=chart_df["Close"],
    name="OHLC",
    increasing=dict(line=dict(color="#16a34a", width=1), fillcolor="#16a34a"),
    decreasing=dict(line=dict(color="#dc2626", width=1), fillcolor="#dc2626"),
    whiskerwidth=0.4,
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"], y=chart_df["MA20"],
    mode="lines", name="MA 20",
    line=dict(color="#6366f1", width=1.5, dash="dot"),
    hovertemplate=f"%{{y:.2f}}<extra>MA 20</extra>",
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"], y=chart_df["MA50"],
    mode="lines", name="MA 50",
    line=dict(color="#f59e0b", width=1.5, dash="dot"),
    hovertemplate=f"%{{y:.2f}}<extra>MA 50</extra>",
))

fig.update_layout(
    height=400,
    margin=dict(l=0, r=0, t=8, b=0),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="DM Sans", size=11, color="#6b7280"),
    xaxis=dict(
        showgrid=False, zeroline=False, showline=False,
        tickfont=dict(size=11, color="#9ca3af"),
        rangeslider=dict(visible=False),
        type="date",
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=11, color="#9ca3af", family="DM Mono"),
        tickprefix=currency,
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
        font=dict(size=11, color="#6b7280"),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#0d1117", bordercolor="#1f2937",
        font=dict(color="#f9fafb", size=12, family="DM Mono"),
    ),
)
fig.update_xaxes(showspikes=True, spikecolor="#d1d5db", spikethickness=1, spikedash="solid")
fig.update_yaxes(showspikes=True, spikecolor="#d1d5db", spikethickness=1, spikedash="solid")

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)


# ── Model Training ───────────────────────────────────────────────────────────
model_df = df[["Open", "High", "Low", "Close"]].dropna().copy()
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
<div class='trained-banner'>
    Model trained &mdash; Linear Regression ready for inference.
</div>
""", unsafe_allow_html=True)


# ── Prediction ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Price Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='section-desc'>Enter Open, High, and Low values to estimate the closing price.</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    open_i = st.number_input(f"Open Price ({currency})", value=round(last_open, 2), step=0.5)
with col2:
    high_i = st.number_input(f"High Price ({currency})", value=round(last_high, 2), step=0.5)
with col3:
    low_i  = st.number_input(f"Low Price ({currency})",  value=round(last_low,  2), step=0.5)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("Run Prediction"):
    try:
        pred  = model.predict([[open_i, high_i, low_i]])
        value = float(pred[0])
        diff  = value - last_price
        direction = f"+{diff:.2f}" if diff >= 0 else f"{diff:.2f}"
        color_hex = "#16a34a" if diff >= 0 else "#dc2626"
        trend     = "Above" if diff >= 0 else "Below"

        st.markdown(f"""
        <div class='predict-result'>
            <div class='predict-label'>Predicted Closing Price</div>
            <div class='predict-value'>{currency}{value:.2f}</div>
            <div class='predict-diff'>
                {trend} current price by &nbsp;
                <span style='color:{color_hex}; font-weight:700; font-family:"DM Mono",monospace;'>{currency}{direction}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
