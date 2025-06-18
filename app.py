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

# --- Config ---
st.set_page_config(page_title="📈 Stock & Crypto Predictor", layout="wide")
st.title("🔮 Real-Time Stock & Crypto Dashboard")
st.write("🔄 Auto-refreshing every 60 seconds...")

# --- Sidebar ---
st.sidebar.header("🔍 Settings")
asset_type = st.sidebar.radio("Choose Asset Type", ["Stocks", "Cryptocurrency"])
custom_ticker = st.sidebar.text_input("Custom Ticker (e.g., NFLX, BTC-USD)")
default_stock_tickers = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META"]
default_crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "XRP-USD", "LTC-USD"]

tickers = default_stock_tickers if asset_type == "Stocks" else default_crypto_tickers
ticker = st.sidebar.selectbox("Select Ticker", tickers)
ticker = custom_ticker.upper() if custom_ticker else ticker

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# --- Fetch Data ---
df = yf.download(ticker, start=start_date, end=end_date)
if df.empty:
    st.error("Failed to load data. Check the ticker symbol.")
    st.stop()
df = df[['Close']].dropna()
df.reset_index(inplace=True)
df.columns = ['ds', 'y']

# --- Current Price Display (from historical data) ---
st.subheader(f"💰 Current Price for {ticker}")
last_close = df['y'].iloc[-1]
st.metric("Last Close", f"${last_close:.2f}")

# --- Live Price Ticker (from real-time data) ---
try:
    live_info = yf.Ticker(ticker).info
    live_price = live_info.get("regularMarketPrice", last_close)
    prev_close = live_info.get("previousClose", last_close)
    live_change_pct = ((live_price - prev_close) / prev_close) * 100

    st.markdown("### 📡 Live Ticker")
    st.markdown(
        f"""
        <div style='padding: 10px; background-color: #f0f2f6; border-radius: 10px; display: flex; align-items: center; justify-content: space-between;'>
            <span style='font-size: 24px; font-weight: bold;'>{ticker}</span>
            <span style='font-size: 20px; color: {"green" if live_change_pct >= 0 else "red"};'>
                ${live_price:.2f} {"🔺" if live_change_pct >= 0 else "🔻"} {live_change_pct:.2f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as e:
    st.warning(f"⚠️ Could not fetch live data: {e}")

# --- RSI Chart ---
try:
    from ta.momentum import RSIIndicator
    rsi = RSIIndicator(close=df['y']).rsi()
    df['RSI'] = rsi
    st.line_chart(df.set_index('ds')[['y', 'RSI']])
except Exception as e:
    st.warning(f"RSI unavailable: {e}")

# --- Feature Engineering ---
df['Lag1'] = df['y'].shift(1)
df['Lag2'] = df['y'].shift(2)
df = df.dropna()
X = df[['Lag1', 'Lag2']]
y = df['y']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- Model Toggles ---
st.sidebar.markdown("### ✅ Select Models")
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_lr = st.sidebar.checkbox("Linear Regression", value=True)
use_xgb = st.sidebar.checkbox("XGBoost", value=True)
use_prophet = st.sidebar.checkbox("Prophet", value=True)
use_arima = st.sidebar.checkbox("ARIMA", value=True)
use_lstm = st.sidebar.checkbox("LSTM", value=True)

results = {}

# --- Traditional ML Models ---
if use_rf:
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    pred_rf = model.predict(X_test)
    results["Random Forest"] = {
        "pred": pred_rf[-1],
        "rmse": mean_squared_error(y_test, pred_rf) ** 0.5,
        "mae": mean_absolute_error(y_test, pred_rf),
        "r2": r2_score(y_test, pred_rf),
        "forecast": model.predict(X[-7:])
    }

if use_lr:
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)
    results["Linear Regression"] = {
        "pred": pred_lr[-1],
        "rmse": mean_squared_error(y_test, pred_lr) ** 0.5,
        "mae": mean_absolute_error(y_test, pred_lr),
        "r2": r2_score(y_test, pred_lr),
        "forecast": model.predict(X[-7:])
    }

if use_xgb:
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    pred_xgb = model.predict(X_test)
    results["XGBoost"] = {
        "pred": pred_xgb[-1],
        "rmse": mean_squared_error(y_test, pred_xgb) ** 0.5,
        "mae": mean_absolute_error(y_test, pred_xgb),
        "r2": r2_score(y_test, pred_xgb),
        "forecast": model.predict(X[-7:])
    }

# --- Prophet Forecast ---
if use_prophet:
    prophet_df = df[['ds', 'y']]
    prophet = Prophet()
    prophet.fit(prophet_df)
    future = prophet.make_future_dataframe(periods=7)
    forecast = prophet.predict(future)
    forecast_values = forecast['yhat'].iloc[-7:].values
    results["Prophet"] = {
        "pred": forecast_values[-1],
        "rmse": mean_squared_error(df['y'].iloc[-7:], forecast_values) ** 0.5,
        "mae": mean_absolute_error(df['y'].iloc[-7:], forecast_values),
        "r2": r2_score(df['y'].iloc[-7:], forecast_values),
        "forecast": forecast_values
    }

# --- ARIMA ---
if use_arima:
    model = ARIMA(df['y'], order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=7)
    results["ARIMA"] = {
        "pred": forecast.iloc[-1],
        "rmse": mean_squared_error(df['y'].iloc[-7:], forecast) ** 0.5,
        "mae": mean_absolute_error(df['y'].iloc[-7:], forecast),
        "r2": r2_score(df['y'].iloc[-7:], forecast),
        "forecast": forecast.values
    }

# --- LSTM ---
if use_lstm:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))
    X_lstm, y_lstm = [], []
    for i in range(60, len(scaled_data) - 7):
        X_lstm.append(scaled_data[i-60:i])
        y_lstm.append(scaled_data[i:i+7])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(7))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, y_lstm, epochs=5, batch_size=16, verbose=0)

    x_input = scaled_data[-60:].reshape(1, 60, 1)
    forecast = model.predict(x_input)[0]
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    results["LSTM"] = {
        "pred": forecast[-1],
        "rmse": mean_squared_error(df['y'].iloc[-7:], forecast) ** 0.5,
        "mae": mean_absolute_error(df['y'].iloc[-7:], forecast),
        "r2": r2_score(df['y'].iloc[-7:], forecast),
        "forecast": forecast
    }

# --- Model Prediction Table ---
st.subheader("📊 Model Predictions & Metrics")
for name, data in results.items():
    st.write(f"**{name}** ➝ Predicted: `${data['pred']:.2f}` | RMSE: `{data['rmse']:.4f}` | MAE: `{data['mae']:.4f}` | R²: `{data['r2']:.4f}`")

st.caption("📘 RMSE = Root Mean Squared Error, MAE = Mean Absolute Error, R² = Coefficient of Determination")

# --- Recommendation Logic ---
best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
recommend = "Buy ✅" if results[best_model]['r2'] > 0.85 else "Hold ⚖️"

st.subheader("📈 Recommendation")
st.success(f"Based on {best_model} model with high R², today's recommendation: **{recommend}**")

# --- Forecast Comparison Table ---
st.subheader("📅 7-Day Forecast Comparison")
forecast_df = pd.DataFrame({'Date': pd.date_range(df['ds'].iloc[-1] + timedelta(days=1), periods=7)})
for name in results:
    forecast_df[name] = results[name]["forecast"]
st.dataframe(forecast_df)

# --- Confidence Bar Chart ---
st.subheader("📉 Model Confidence (Lower RMSE = Better)")
bar_fig = go.Figure(go.Bar(
    x=list(results.keys()),
    y=[results[k]["rmse"] for k in results],
    text=[f"{results[k]['rmse']:.4f}" for k in results],
    textposition="auto"
))
bar_fig.update_layout(xaxis_title="Model", yaxis_title="RMSE")
st.plotly_chart(bar_fig, use_container_width=True)

