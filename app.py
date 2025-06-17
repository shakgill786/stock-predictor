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

default_tickers = {
    "Stocks": ["AAPL", "TSLA", "GOOG", "MSFT", "AMZN", "NVDA", "META"],
    "Cryptocurrency": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "XRP-USD", "BNB-USD"]
}

tickers = default_tickers[asset_type]
ticker = st.sidebar.selectbox("Select Ticker", tickers)
custom_ticker = st.sidebar.text_input("Or enter custom ticker (e.g. NFLX, LTC-USD)", "")
ticker = custom_ticker.strip().upper() if custom_ticker else ticker

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# --- Model Toggles ---
st.sidebar.markdown("### ✅ Toggle Models")
enabled_models = {
    "Random Forest": st.sidebar.checkbox("Random Forest", value=True),
    "Linear Regression": st.sidebar.checkbox("Linear Regression", value=True),
    "XGBoost": st.sidebar.checkbox("XGBoost", value=True),
    "Prophet": st.sidebar.checkbox("Prophet", value=True),
    "ARIMA": st.sidebar.checkbox("ARIMA", value=True),
    "LSTM": st.sidebar.checkbox("LSTM", value=True)
}

# --- Fetch Data ---
df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']].dropna()
df.reset_index(inplace=True)
df.columns = ['ds', 'y']

if df.empty:
    st.error("⚠️ No data available. Please check the ticker or date range.")
    st.stop()

# --- Current Price ---
st.subheader(f"💰 Current Price for {ticker}")
latest_price = df['y'].iloc[-1]
st.metric("Last Close", f"${latest_price:.2f}")

# --- Feature Engineering ---
df['Lag1'] = df['y'].shift(1)
df['Lag2'] = df['y'].shift(2)
df = df.dropna()
X = df[['Lag1', 'Lag2']]
y = df['y']
X_train, X_test = X[:-7], X[-7:]
y_train, y_test = y[:-7], y[-7:]

# --- Model Forecasts ---
results = {}
actual_7 = y_test.values

# --- Traditional ML Models ---
ml_models = {
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100)
}

for name, model in ml_models.items():
    if not enabled_models[name]: continue
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
    results[name] = {
        "forecast": forecast,
        "pred": forecast[-1],
        "rmse": mean_squared_error(actual_7, forecast) ** 0.5,
        "mae": mean_absolute_error(actual_7, forecast),
        "r2": r2_score(actual_7, forecast)
    }

# --- Prophet ---
if enabled_models["Prophet"]:
    df_prophet = df[['ds', 'y']]
    prophet = Prophet()
    prophet.fit(df_prophet)
    future = prophet.make_future_dataframe(periods=7)
    forecast_df = prophet.predict(future)
    forecast = forecast_df['yhat'].iloc[-7:].values
    results["Prophet"] = {
        "forecast": forecast,
        "pred": forecast[-1],
        "rmse": mean_squared_error(actual_7, forecast) ** 0.5,
        "mae": mean_absolute_error(actual_7, forecast),
        "r2": r2_score(actual_7, forecast)
    }

# --- ARIMA ---
if enabled_models["ARIMA"]:
    arima_model = ARIMA(df['y'], order=(5,1,0)).fit()
    forecast = arima_model.forecast(steps=7)
    results["ARIMA"] = {
        "forecast": forecast.values,
        "pred": forecast.iloc[-1],
        "rmse": mean_squared_error(actual_7, forecast) ** 0.5,
        "mae": mean_absolute_error(actual_7, forecast),
        "r2": r2_score(actual_7, forecast)
    }

# --- LSTM ---
if enabled_models["LSTM"]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df['y'].values.reshape(-1, 1))
    X_lstm, y_lstm = [], []
    for i in range(60, len(scaled) - 7):
        X_lstm.append(scaled[i-60:i])
        y_lstm.append(scaled[i:i+7])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model_lstm.add(LSTM(50))
    model_lstm.add(Dense(7))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=16, verbose=0)

    x_input = scaled[-60:].reshape(1, 60, 1)
    scaled_forecast = model_lstm.predict(x_input)[0]
    forecast = scaler.inverse_transform(scaled_forecast.reshape(-1, 1)).flatten()

    results["LSTM"] = {
        "forecast": forecast,
        "pred": forecast[-1],
        "rmse": mean_squared_error(actual_7, forecast) ** 0.5,
        "mae": mean_absolute_error(actual_7, forecast),
        "r2": r2_score(actual_7, forecast)
    }

# --- Prediction Metrics ---
st.subheader("📊 Model Predictions")
for name, res in results.items():
    st.markdown(
        f"**{name}** ➝ "
        f"Pred: `${res['pred']:.2f}` | "
        f"RMSE: `{res['rmse']:.4f}` | "
        f"MAE: `{res['mae']:.4f}` | "
        f"R²: `{res['r2']:.4f}`"
    )

# --- Forecast Comparison Table ---
st.subheader("📅 7-Day Forecast Comparison")
comparison_df = pd.DataFrame({"Date": df['ds'].iloc[-7:].values, "Actual": actual_7})
for name, res in results.items():
    comparison_df[name] = res['forecast']
st.dataframe(comparison_df)

# --- Comparison Chart ---
st.subheader("📈 7-Day Prediction vs Actual Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=comparison_df["Date"], y=comparison_df["Actual"], name="Actual", line=dict(width=3)))
for name in results:
    fig.add_trace(go.Scatter(x=comparison_df["Date"], y=comparison_df[name], name=name))
fig.update_layout(xaxis_title="Date", yaxis_title="Price", legend_title="Model")
st.plotly_chart(fig, use_container_width=True)

# --- Confidence Bar Chart ---
st.subheader("📉 Model Confidence (Error Comparison)")
bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(
    x=list(results.keys()),
    y=[results[name]['rmse'] for name in results],
    name="RMSE",
    text=[f"{results[name]['rmse']:.4f}" for name in results],
    textposition="auto"
))
bar_fig.add_trace(go.Bar(
    x=list(results.keys()),
    y=[results[name]['mae'] for name in results],
    name="MAE",
    text=[f"{results[name]['mae']:.4f}" for name in results],
    textposition="auto"
))
bar_fig.update_layout(barmode='group', yaxis_title="Error", xaxis_title="Model")
st.plotly_chart(bar_fig, use_container_width=True)
