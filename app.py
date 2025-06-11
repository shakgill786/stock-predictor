import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ta.momentum import RSIIndicator
from prophet import Prophet

# our custom helpers
from models.lstm_model  import train_lstm_model, predict_lstm
from models.arima_model import train_arima_model, predict_arima, evaluate_arima

# ───────────────────────────────  UI SET-UP  ────────────────────────────────
st.set_page_config(page_title="📈 Market Predictor", layout="wide")
st.title("🔮 Stock & Crypto Market Predictor")

st.sidebar.header("Settings")
asset_type = st.sidebar.selectbox("Asset Type", ["Stocks", "Cryptocurrency"])
tickers = (["AAPL", "TSLA", "MSFT", "GOOG", "META", "NVDA"]
           if asset_type == "Stocks"
           else ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "ADA-USD"])
default_tkr  = st.sidebar.selectbox("Choose a Ticker", tickers)
custom_tkr   = st.sidebar.text_input("…or type a custom ticker")
ticker       = custom_tkr.upper().strip() or default_tkr

search_btn   = st.sidebar.button("🔍 Run")
refresh_sec  = st.sidebar.slider("Auto-refresh (sec)", 30, 300, 60)

start_date = datetime.today() - timedelta(days=365)
end_date   = datetime.today()

# ───────────────────────────────  MAIN LOGIC  ───────────────────────────────
if search_btn:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error("❌ No data found for that ticker.")
            st.stop()

        # ── Price & Indicators ──────────────────────────────────────────────
        st.subheader(f"📊 {ticker} Price History")
        st.line_chart(df["Close"])

        df["SMA_50"]  = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        try:
            df["RSI"] = RSIIndicator(df["Close"]).rsi()
        except Exception:
            df["RSI"] = np.nan

        st.subheader("📈 Moving Averages (SMA 50 & 200)")
        st.line_chart(df[["Close", "SMA_50", "SMA_200"]].dropna())

        if df["RSI"].notna().any():
            st.subheader("📉 RSI")
            st.line_chart(df["RSI"].dropna())

        # ── Prepare Features ───────────────────────────────────────────────
        df["Tomorrow_Close"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        if df.shape[0] < 20:
            st.warning("Not enough data for ML.")
            st.stop()

        X = df[["Open", "High", "Low", "Close", "Volume"]]
        y = df["Tomorrow_Close"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # ── Classic ML Models ──────────────────────────────────────────────
        classic_models = {
            "Random Forest":  RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost":        XGBRegressor(n_estimators=100, random_state=42),
            "Linear Regression": LinearRegression()
        }
        comparison = []          # (name, rmse)
        next_preds = {}          # 1-step ahead
        past7_preds = {}         # predictions for last 7 actual closes

        for name, mdl in classic_models.items():
            mdl.fit(X_train, y_train)
            rmse = mean_squared_error(y_test, mdl.predict(X_test), squared=False)
            comparison.append((name, rmse))

            # next-day prediction
            next_preds[name] = mdl.predict(X.iloc[[-1]])[0]

            # predictions for previous 7 actual closes
            past7_preds[name] = mdl.predict(X.tail(7))

        # ── LSTM ───────────────────────────────────────────────────────────
        lstm_mdl, lstm_scaler, X_lstm, y_lstm = train_lstm_model(df["Close"])
        lstm_rmse = mean_squared_error(
            y_lstm[-len(X_test):], lstm_mdl.predict(X_lstm[-len(X_test):]).flatten(), squared=False
        )
        comparison.append(("LSTM", lstm_rmse))
        next_preds["LSTM"]  = predict_lstm(lstm_mdl, lstm_scaler, df["Close"], steps=1)[0]
        past7_preds["LSTM"] = predict_lstm(lstm_mdl, lstm_scaler, df["Close"], steps=7)

        # ── ARIMA ──────────────────────────────────────────────────────────
        arima_fit   = train_arima_model(df["Close"])
        arima_rmse  = evaluate_arima(arima_fit, df["Close"])
        comparison.append(("ARIMA", arima_rmse))
        next_preds["ARIMA"]  = predict_arima(arima_fit, steps=1)[0]
        past7_preds["ARIMA"] = arima_fit.predict(start=len(df)-7, end=len(df)-1).values

        # ── Confidence Bars (all 5) ────────────────────────────────────────
        st.subheader("📊 Model Confidence (lower RMSE ⇒ higher confidence)")
        max_rmse = max(r for _, r in comparison)
        min_rmse = min(r for _, r in comparison)
        for name, rmse in sorted(comparison, key=lambda x: x[1]):
            conf = 1 - (rmse - min_rmse) / (max_rmse - min_rmse + 1e-9)
            st.progress(conf, text=f"{name}: {conf*100:.1f}% | RMSE {rmse:.2f}")

        # ── Next-day Prediction Table ─────────────────────────────────────
        st.subheader("📅 Next-Day Predictions")
        st.table(pd.Series(next_preds, name="Predicted Close $").round(2))

        # ── Live Price ────────────────────────────────────────────────────
        st.subheader("💰 Live Price")
        try:
            info  = yf.Ticker(ticker)
            live  = info.fast_info.get("lastPrice") or info.info.get("regularMarketPrice")
            prev  = info.fast_info.get("previousClose") or info.info.get("previousClose")
            delta = f"{live-prev:+.2f}" if live and prev else None
            col1, col2 = st.columns([2,1])
            col1.metric("Current Price", f"${live:.2f}", delta)
            col2.caption(datetime.now().strftime("Updated %Y-%m-%d %H:%M:%S"))
        except Exception as e:
            st.warning(f"Live price error: {e}")

        # ── Prophet Forecast ──────────────────────────────────────────────
        st.subheader("🔮 Prophet 7-Day Forecast")
        try:
            prop_df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds","Close":"y"})
            model_p = Prophet(daily_seasonality=True)
            model_p.fit(prop_df)
            future  = model_p.make_future_dataframe(periods=7)
            fcst    = model_p.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prop_df["ds"], y=prop_df["y"],
                                     name="Historical", line=dict(color="gray")))
            fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"],
                                     name="Forecast", line=dict(color="orange")))
            fig.update_layout(xaxis_title="Date", yaxis_title="Price",
                              title=f"{ticker} Prophet Forecast")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                fcst[["ds","yhat"]].tail(7)
                    .rename(columns={"ds":"Date","yhat":"Forecasted Close"})
                    .round(2)
            )
        except Exception as e:
            st.warning(f"Prophet error: {e}")

        # ── 7-Day Actual vs. Predicted Table ──────────────────────────────
        st.subheader("📈 Past 7 Days: Actual vs All Models")
        last7_dates   = df.index[-7:]
        compare_table = pd.DataFrame({"Date": last7_dates.date,
                                      "Actual Close": df["Close"].tail(7).values})

        # add predictions
        for name, preds in past7_preds.items():
            compare_table[name] = np.array(preds).flatten()

        st.dataframe(compare_table.set_index("Date").round(2))

        # ── Auto-refresh Loop ─────────────────────────────────────────────
        st.caption(f"🔄 Auto-refreshing every {refresh_sec} seconds…")
        time.sleep(refresh_sec)
        st.rerun()

    except Exception as err:
        st.error(f"Unhandled error: {err}")

else:
    st.info("👈 Select a ticker and click 'Run' to begin.")
