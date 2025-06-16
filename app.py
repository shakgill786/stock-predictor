import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
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
option = st.sidebar.radio("Choose Asset Type", ["Stocks", "Cryptocurrency"])

if option == "Stocks":
    tickers = ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META"]
else:
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]

ticker = st.sidebar.selectbox("Select Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# --- Fetch Data ---
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']].dropna()
df.reset_index(inplace=True)
df.columns = ['ds', 'y']

# Show current price
st.subheader(f"💰 Current Price for {ticker}")
latest_price = df['y'].iloc[-1]
st.metric("Last Close", f"${latest_price:.2f}")

# --- Feature Engineering for ML Models ---
df['Lag1'] = df['y'].shift(1)
df['Lag2'] = df['y'].shift(2)
df = df.dropna()
X = df[['Lag1', 'Lag2']]
y = df['y']

# --- Train/Test Split ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- Traditional ML Models ---
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    results[name] = {
        "pred": y_pred[-1],
        "rmse": rmse,
        "forecast": model.predict(X[-7:])
    }

# --- Prophet Forecast ---
df_prophet = df[['ds', 'y']]
prophet = Prophet()
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=7)
forecast_prophet = prophet.predict(future)
results["Prophet"] = {
    "pred": forecast_prophet['yhat'].iloc[-1],
    "rmse": mean_squared_error(df['y'].iloc[-7:], forecast_prophet['yhat'].iloc[-14:-7]) ** 0.5,
    "forecast": forecast_prophet['yhat'].iloc[-7:].values
}

# --- ARIMA ---
arima_model = ARIMA(df['y'], order=(5,1,0)).fit()
forecast_arima = arima_model.forecast(steps=7)
results["ARIMA"] = {
    "pred": forecast_arima.iloc[-1],
    "rmse": mean_squared_error(df['y'].iloc[-7:], forecast_arima) ** 0.5,
    "forecast": forecast_arima.values
}

# --- LSTM ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))
X_lstm, y_lstm = [], []
for i in range(60, len(scaled_data) - 7):
    X_lstm.append(scaled_data[i-60:i])
    y_lstm.append(scaled_data[i:i+7])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(7))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=16, verbose=0)

x_input = scaled_data[-60:].reshape(1, 60, 1)
forecast_lstm_scaled = model_lstm.predict(x_input)[0]
forecast_lstm = scaler.inverse_transform(forecast_lstm_scaled.reshape(-1, 1)).flatten()
results["LSTM"] = {
    "pred": forecast_lstm[-1],
    "rmse": mean_squared_error(df['y'].iloc[-7:], forecast_lstm) ** 0.5,
    "forecast": forecast_lstm
}

# --- Predictions Table ---
st.subheader("📊 Model Predictions")
for name, data in results.items():
    st.write(f"**{name}** ➝ Predicted Price: `${data['pred']:.2f}` | RMSE: `{data['rmse']:.4f}`")

# --- 7-Day Forecast Comparison Table ---
st.subheader("📅 7-Day Forecast Comparison")
forecast_df = pd.DataFrame({'Date': pd.date_range(df['ds'].iloc[-1] + timedelta(days=1), periods=7)})
for name in results:
    forecast_df[name] = results[name]["forecast"]
st.dataframe(forecast_df)

# --- Confidence Bar Chart ---
st.subheader("📉 Model Confidence (RMSE Comparison)")
bar_fig = go.Figure(go.Bar(
    x=[name for name in results],
    y=[results[name]['rmse'] for name in results],
    text=[f"{results[name]['rmse']:.4f}" for name in results],
    textposition="auto"
))
bar_fig.update_layout(yaxis_title="RMSE (Lower = Better)", xaxis_title="Model")
st.plotly_chart(bar_fig, use_container_width=True)
