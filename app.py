import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.momentum import RSIIndicator
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("🔮 Stock Market Dashboard")

# --- Sidebar ---
st.sidebar.header("Settings")
tickers = ["AAPL", "TSLA", "MSFT", "GOOG", "NVDA"]
default_ticker = st.sidebar.selectbox("Choose a Popular Ticker", tickers)
custom_ticker = st.sidebar.text_input("Or enter a custom ticker (overrides selection)", "")
search_button = st.sidebar.button("🔍 Search")

# --- Date Range ---
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# --- Ticker Logic ---
ticker = custom_ticker.upper().strip() if custom_ticker else default_ticker

# --- Load Data ---
if search_button or not custom_ticker:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)

        if df.empty:
            st.error("❌ No data found for that ticker. Try another one.")
        else:
            # --- Flatten MultiIndex ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            st.subheader(f"📊 Price Data for {ticker}")
            st.line_chart(df["Close"])

            # --- Moving Averages ---
            df["SMA_50"] = df["Close"].rolling(window=50).mean()
            df["SMA_200"] = df["Close"].rolling(window=200).mean()

            try:
                rsi = RSIIndicator(close=df["Close"].astype(float).squeeze(), window=14)
                df["RSI"] = rsi.rsi()
            except Exception as e:
                st.warning(f"RSI calc error: {e}")
                df["RSI"] = np.nan

            # --- MA Chart ---
            ma_df = df[["Close", "SMA_50", "SMA_200"]].dropna()
            if not ma_df.empty:
                st.subheader("📈 Moving Averages")
                st.line_chart(ma_df)
            else:
                st.warning("📉 Not enough data to display moving averages.")

            # --- RSI Chart ---
            if df["RSI"].dropna().shape[0] > 0:
                st.subheader("📉 RSI")
                st.line_chart(df["RSI"])
            else:
                st.warning("📉 Not enough RSI data.")

            # --- ML Predictions ---
            df["Tomorrow_Close"] = df["Close"].shift(-1)
            df.dropna(inplace=True)

            st.markdown(f"### 🔢 Cleaned Data: `{df.shape[0]}` rows")
            st.dataframe(df.tail())

            if df.shape[0] < 20:
                st.warning("⚠️ Not enough data for ML predictions.")
            else:
                X = df[["Open", "High", "Low", "Close", "Volume"]]
                y = df["Tomorrow_Close"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                result_df = pd.DataFrame({
                    "Date": y_test.index,
                    "Actual": y_test.values,
                    "Predicted": preds
                }).sort_values(by="Date")
                result_df.set_index("Date", inplace=True)

                st.subheader("🧠 ML Predictions")
                st.line_chart(result_df)

                latest_data = X.iloc[[-1]]
                next_day_prediction = model.predict(latest_data)[0]
                st.metric(label=f"📌 Predicted Next Close ({ticker})", value=f"${next_day_prediction:.2f}")

                rmse = mean_squared_error(y_test, preds, squared=False)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                st.markdown("### 📉 Model Performance")
                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**R² Score:** {r2:.4f}")

                with st.expander("📋 Prediction Data"):
                    st.dataframe(result_df.reset_index().tail(20))

            # --- Prophet Forecasting ---
            st.subheader("🔮 7-Day Forecast")
            forecast_df = df.reset_index()[["Date", "Close"]].dropna()
            forecast_df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

            if forecast_df.shape[0] < 30:
                st.warning("⚠️ Not enough data to forecast future prices.")
            else:
                prophet_model = Prophet(daily_seasonality=True)
                prophet_model.fit(forecast_df)

                future = prophet_model.make_future_dataframe(periods=7)
                forecast = prophet_model.predict(future)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
                fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["y"], name="Historical"))
                fig.update_layout(
                    title="📅 Prophet Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
