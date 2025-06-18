import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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

# --- Page config and auto-refresh ---
st.set_page_config(page_title="📈 Stock & Crypto Predictor", layout="wide")
st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

# --- Sidebar settings ---
st.sidebar.header("🔍 Settings")
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Cryptocurrency"])
custom_ticker = st.sidebar.text_input("Custom Ticker", "")
defaults = {
    "Stocks": ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META"],
    "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "XRP-USD", "LTC-USD"]
}
tickers = defaults[asset_type]
selected = st.sidebar.selectbox("Select Ticker", tickers)
ticker = custom_ticker.strip().upper() or selected
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# --- Model selection ---
st.sidebar.markdown("### Models to run")
model_flags = {
    "Random Forest": st.sidebar.checkbox("Random Forest", True),
    "Linear Regression": st.sidebar.checkbox("Linear Regression", True),
    "XGBoost": st.sidebar.checkbox("XGBoost", True),
    "Prophet": st.sidebar.checkbox("Prophet", True),
    "ARIMA": st.sidebar.checkbox("ARIMA", True),
    "LSTM": st.sidebar.checkbox("LSTM", True),
}

# --- Data loader with caching ---
@st.cache_data(ttl=3600)
def fetch_close_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    df = df[["Close"]].dropna()
    df.reset_index(inplace=True)
    df.columns = ["ds", "y"]
    return df

# --- Fetch and validate ---
try:
    df = fetch_close_data(ticker, start_date, end_date)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

# --- Header and current price ---
st.title(f"🔮 {asset_type} Predictor & Dashboard")
current_price = df['y'].iloc[-1]
st.metric(f"Current Close for {ticker}", f"${current_price:.2f}")

# --- Live price (best-effort) ---
try:
    info = yf.Ticker(ticker).info
    live = info.get("regularMarketPrice", current_price)
    prev = info.get("previousClose", current_price)
    pct = (live - prev) / prev * 100
    col1, col2 = st.columns([1, 3])
    col1.markdown(f"<h3>{ticker}</h3>", unsafe_allow_html=True)
    col2.markdown(
        f"<p style='font-size:24px;color:{'green' if pct>=0 else 'red'};'>${live:.2f} {'▲' if pct>=0 else '▼'} {pct:.2f}%</p>",
        unsafe_allow_html=True
    )
except Exception:
    st.warning("⚠️ Live price unavailable.")

# --- Compute RSI ---
try:
    from ta.momentum import RSIIndicator
    df['RSI'] = RSIIndicator(close=df['y']).rsi()
    st.subheader("📈 Price & RSI")
    st.line_chart(df.set_index('ds')[['y', 'RSI']])
except Exception:
    st.warning("RSI calculation unavailable.")

# --- Feature engineering ---
df['Lag1'] = df['y'].shift(1)
df['Lag2'] = df['y'].shift(2)
df.dropna(inplace=True)
X = df[['Lag1', 'Lag2']]
y = df['y']
split = int(len(df)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Train and forecast ---
results = {}

# Base ML loop
def evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    last_pred = model.predict(X.tail(7))
    results[name] = {
        'pred': preds[-1],
        'rmse': mean_squared_error(y_test, preds, squared=False),
        'mae': mean_absolute_error(y_test, preds),
        'r2': r2_score(y_test, preds),
        'forecast': last_pred
    }

if model_flags['Random Forest']:
    evaluate(RandomForestRegressor(n_estimators=100), 'Random Forest')
if model_flags['Linear Regression']:
    evaluate(LinearRegression(), 'Linear Regression')
if model_flags['XGBoost']:
    evaluate(XGBRegressor(objective='reg:squarederror', n_estimators=100), 'XGBoost')

# Prophet
if model_flags['Prophet']:
    pr = Prophet()
    pr.fit(df[['ds','y']])
    fut = pr.make_future_dataframe(periods=7)
    fc = pr.predict(fut)['yhat']
    last7 = fc.iloc[-7:].values
    actual7 = df['y'].iloc[-7:].values
    results['Prophet'] = {
        'pred': last7[-1],
        'rmse': mean_squared_error(actual7, last7, squared=False),
        'mae': mean_absolute_error(actual7, last7),
        'r2': r2_score(actual7, last7),
        'forecast': last7
    }

# ARIMA\***
if model_flags['ARIMA']:
    ar = ARIMA(df['y'], order=(5,1,0)).fit()
    fc = ar.forecast(7)
    actual7 = df['y'].iloc[-7:].values
    results['ARIMA'] = {
        'pred': fc.iloc[-1],
        'rmse': mean_squared_error(actual7, fc, squared=False),
        'mae': mean_absolute_error(actual7, fc),
        'r2': r2_score(actual7, fc),
        'forecast': fc.values
    }

# LSTM\***
if model_flags['LSTM']:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['y'].values.reshape(-1,1))
    seq, target = [], []
    for i in range(60, len(data)-7):
        seq.append(data[i-60:i])
        target.append(data[i:i+7].flatten())
    Xl, yl = np.array(seq), np.array(target)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        LSTM(50),
        Dense(7)
    ])
    model.compile('adam', 'mse')
    model.fit(Xl, yl, epochs=5, batch_size=16, verbose=0)
    inp = data[-60:].reshape(1,60,1)
    fc = model.predict(inp)[0]
    fc = scaler.inverse_transform(fc.reshape(-1,1)).flatten()
    actual7 = df['y'].iloc[-7:].values
    results['LSTM'] = {
        'pred': fc[-1],
        'rmse': mean_squared_error(actual7, fc, squared=False),
        'mae': mean_absolute_error(actual7, fc),
        'r2': r2_score(actual7, fc),
        'forecast': fc
    }

# --- Display predictions ---
st.subheader("📊 Model Predictions & Metrics")
for m, res in results.items():
    st.write(f"**{m}**: Pred=${res['pred']:.2f} | RMSE={res['rmse']:.4f} | MAE={res['mae']:.4f} | R²={res['r2']:.4f}")

# Recommendation
best = min(results, key=lambda k: results[k]['rmse'])
rec = "Buy ✅" if results[best]['r2']>0.85 else "Hold ⚖️"
st.subheader("📈 Recommendation")
st.success(f"Based on {best}, recommendation: {rec}")

# 7-day forecast table and chart
st.subheader("📅 7-Day Forecast")
fdf = pd.DataFrame({ 'Date': pd.date_range(df['ds'].iloc[-1]+timedelta(1), periods=7) })
for m in results:
    fdf[m] = results[m]['forecast']
st.dataframe(fdf)

st.subheader("📉 Actual vs Predicted (Last 7 Days)")
last_dates = df['ds'].iloc[-7:]
actuals = df['y'].iloc[-7:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=last_dates, y=actuals, mode='lines+markers', name='Actual', line=dict(width=3)))
for m in results:
    fig.add_trace(go.Scatter(x=last_dates, y=results[m]['forecast'], mode='lines+markers', name=m))
fig.update_layout(title='Actual vs Forecast', xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Model Confidence (RMSE)")
bar = go.Figure(go.Bar(x=list(results), y=[results[m]['rmse'] for m in results], text=[f"{results[m]['rmse']:.2f}" for m in results], textposition='auto'))
bar.update_layout(xaxis_title='Model', yaxis_title='RMSE')
st.plotly_chart(bar, use_container_width=True)
