import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- Page config & auto-refresh ---
st.set_page_config(page_title="📈 Stock & Crypto Predictor", layout="wide")
# meta-refresh every 60 seconds
st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

# --- Sidebar: settings & models ---
st.sidebar.header("🔍 Settings")
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Cryptocurrency"])
custom_ticker = st.sidebar.text_input("Custom Ticker", "")
defaults = {
    "Stocks": ["AAPL","TSLA","GOOG","MSFT","AMZN","NVDA","META"],
    "Cryptocurrency": ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOGE-USD","XRP-USD","LTC-USD"]
}
tickers = defaults[asset_type]
selected = st.sidebar.selectbox("Select Ticker", tickers)
ticker = custom_ticker.strip().upper() or selected

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

st.sidebar.markdown("### Models to run")
model_flags = {
    "Random Forest":       st.sidebar.checkbox("Random Forest", True),
    "Linear Regression":   st.sidebar.checkbox("Linear Regression", True),
    "XGBoost":             st.sidebar.checkbox("XGBoost", True),
    "Prophet":             st.sidebar.checkbox("Prophet", True),
    "ARIMA":               st.sidebar.checkbox("ARIMA", True),
    "LSTM":                st.sidebar.checkbox("LSTM", True),
}

# --- Data loader with caching ---
@st.cache_data(ttl=3600)
def fetch_close_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Attempts to fetch historical close data via yf.download, with fallback to Ticker.history.
    Raises ValueError on failure.
    """
    end_inc = end + timedelta(days=1)
    st.write(f"🔍 Loading {symbol} from {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    df = None
    # Primary: yf.download
    try:
        df = yf.download(symbol, start=start, end=end_inc, threads=False, progress=False)
        if df is None or df.empty:
            st.write("⚠️ yf.download returned empty; trying Ticker.history...")
            df = None
        else:
            st.write("✅ Data loaded via yf.download")
    except Exception as e:
        st.write(f"❌ yf.download error: {e}")
        df = None

    # Fallback: Ticker.history
    if df is None:
        try:
            st.write("🔍 Fetching via Ticker.history...")
            ticker_obj = yf.Ticker(symbol)
            time.sleep(0.5)
            df = ticker_obj.history(start=start, end=end_inc, interval="1d", actions=False)
            if df is None or df.empty:
                st.write("❌ Ticker.history returned empty")
                df = None
            else:
                st.write("✅ Data loaded via Ticker.history")
        except Exception as e:
            st.write(f"❌ Ticker.history error: {e}")
            df = None

    if df is None or df.empty:
        raise ValueError(f"No data available for {symbol}")

    df = (
        df[["Close"]]
          .rename(columns={"Close":"y"})
          .reset_index()
          .rename(columns={"Date":"ds"})
          .dropna()
    )
    return df

# --- Fetch & validate ---
try:
    df = fetch_close_data(ticker, start_date, end_date)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

# --- Header & current price ---
st.title(f"🔮 {asset_type} Predictor & Dashboard")
current_price = df["y"].iloc[-1]
st.metric(f"Current Close for {ticker}", f"${current_price:.2f}")

# --- Live price snippet ---
try:
    info = yf.Ticker(ticker).info
    live = info.get("regularMarketPrice", current_price)
    prev = info.get("previousClose", current_price)
    pct  = (live - prev) / prev * 100
    c1,c2 = st.columns([1,3])
    c1.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)
    c2.markdown(
        f"<p style='font-size:24px;color:{'green' if pct>=0 else 'red'};'>"
        f"${live:.2f} {'▲' if pct>=0 else '▼'} {pct:.2f}%"
        "</p>",
        unsafe_allow_html=True
    )
except Exception:
    st.warning("⚠️ Live price unavailable.")

# --- RSI plot ---
try:
    from ta.momentum import RSIIndicator
    df["RSI"] = RSIIndicator(close=df["y"]).rsi()
    st.subheader("📈 Price & RSI")
    st.line_chart(df.set_index("ds")[['y','RSI']])
except Exception:
    st.warning("RSI calculation unavailable.")

# --- Feature engineering & train/test split ---
df["Lag1"] = df["y"].shift(1)
df["Lag2"] = df["y"].shift(2)
df.dropna(inplace=True)
X = df[["Lag1","Lag2"]]
y = df["y"]
split = int(len(df)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Run & evaluate models ---
results = {}
def evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    fc7   = model.predict(X.tail(7))
    results[name] = {
        "pred": preds[-1],
        "rmse": mean_squared_error(y_test, preds, squared=False),
        "mae":  mean_absolute_error(y_test, preds),
        "r2":   r2_score(y_test, preds),
        "forecast": fc7
    }

if model_flags["Random Forest"]:
    evaluate(RandomForestRegressor(n_estimators=100), "Random Forest")
if model_flags["Linear Regression"]:
    evaluate(LinearRegression(), "Linear Regression")
if model_flags["XGBoost"]:
    evaluate(XGBRegressor(objective="reg:squarederror", n_estimators=100), "XGBoost")

# --- Prophet ---
if model_flags["Prophet"]:
    m = Prophet()
    m.fit(df[["ds","y"]])
    fut = m.make_future_dataframe(periods=7)
    ph = m.predict(fut)["yhat"].iloc[-7:].values
    act = df["y"].iloc[-7:].values
    results["Prophet"] = {
        "pred": ph[-1],
        "rmse": mean_squared_error(act, ph, squared=False),
        "mae":  mean_absolute_error(act, ph),
        "r2":   r2_score(act, ph),
        "forecast": ph
    }

# --- ARIMA ---
if model_flags["ARIMA"]:
    ar = ARIMA(df["y"], order=(5,1,0)).fit()
    ar_fc = ar.forecast(7)
    act   = df["y"].iloc[-7:].values
    results["ARIMA"] = {
        "pred": ar_fc.iloc[-1],
        "rmse": mean_squared_error(act, ar_fc, squared=False),
        "mae":  mean_absolute_error(act, ar_fc),
        "r2":   r2_score(act, ar_fc),
        "forecast": ar_fc.values
    }

# --- LSTM ---
if model_flags["LSTM"]:
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(df["y"].values.reshape(-1,1))
    seq, tgt = [], []
    for i in range(60, len(vals)-7):
        seq.append(vals[i-60:i])
        tgt.append(vals[i:i+7].flatten())
    Xl, yl = np.array(seq), np.array(tgt)
    lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(7)
    ])
    lstm.compile("adam","mse")
    lstm.fit(Xl, yl, epochs=5, batch_size=16, verbose=0)
    inp = vals[-60:].reshape(1,60,1)
    fc  = lstm.predict(inp)[0]
    fc  = scaler.inverse_transform(fc.reshape(-1,1)).flatten()
    act = df["y"].iloc[-7:].values
    results["LSTM"] = {
        "pred": fc[-1],
        "rmse": mean_squared_error(act, fc, squared=False),
        "mae":  mean_absolute_error(act, fc),
        "r2":   r2_score(act, fc),
        "forecast": fc
    }

# --- Display metrics & recommendation ---
st.subheader("📊 Model Predictions & Metrics")
for name,res in results.items():
    st.write(f"**{name}** – Pred=${{res['pred']:.2f}} | RMSE={{res['rmse']:.4f}} | MAE={{res['mae']:.4f}} | R²={{res['r2']:.4f}}")

best = min(results, key=lambda m: results[m]["rmse"])
rec  = "Buy ✅" if results[best]["r2"] > 0.85 else "Hold ⚖️"
st.subheader("📈 Recommendation")
st.success(f"Based on **{best}**, recommendation: **{rec}**")

# --- 7-day forecast table & charts ---
st.subheader("📅 7-Day Forecast")
fdf = pd.DataFrame({"Date": pd.date_range(df["ds"].iloc[-1]+timedelta(1), periods=7)})
for name in results:
    fdf[name] = results[name]["forecast"]
st.dataframe(fdf)

st.subheader("📉 Actual vs Predicted (Last 7 Days)")
last7 = df["ds"].iloc[-7:]
act7  = df["y"].iloc[-7:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=last7, y=act7, mode="lines+markers", name="Actual", line=dict(width=3)))
for name in results:
    fig.add_trace(go.Scatter(x=last7, y=results[name]["forecast"], mode="lines+markers", name=name))
fig.update_layout(
    title="Actual vs Forecast",
    xaxis_title="Date", yaxis_title="Price ($)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Model Confidence (RMSE)")
bar = go.Figure(go.Bar(
    x=list(results.keys()),
    y=[results[m]["rmse"] for m in results],
    text=[f"{results[m]['rmse']:.2f}" for m in results],
    textposition="auto"
))
bar.update_layout(xaxis_title="Model", yaxis_title="RMSE")
st.plotly_chart(bar, use_container_width=True)
