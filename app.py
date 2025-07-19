import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# --- Streamlit config ---
st.set_page_config(page_title="📈 Stock & Crypto Predictor", layout="wide")
st.title("🔮 Real-Time Stock & Crypto Dashboard")
st.write("🔄 Auto-refreshing every 60 seconds…")

# --- Sidebar settings ---
st.sidebar.header("🔍 Settings")
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Cryptocurrency"])
custom     = st.sidebar.text_input("Custom Ticker (e.g., NFLX, BTC-USD)")
stocks     = ["AAPL","TSLA","GOOG","MSFT","AMZN","NVDA","META"]
cryptos    = ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOGE-USD","XRP-USD","LTC-USD"]
tickers    = stocks if asset_type=="Stocks" else cryptos
ticker     = custom.upper() if custom else st.sidebar.selectbox("Select Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date   = st.sidebar.date_input("End Date",   datetime.now())
horizon    = 7 if st.sidebar.selectbox("Forecast Horizon", ["7-Day","3-Day"])=="7-Day" else 3

# --- Fetch & clean data ---
df = yf.download(ticker, start=start_date, end=end_date, group_by=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(0)
df = df[['Open','High','Low','Close','Volume']].dropna()
if df.empty:
    st.error("❌ No data—check your ticker")
    st.stop()

# --- Current & live price ---
last_close = float(df['Close'].iloc[-1])
st.subheader(f"💰 Current Price for {ticker}")
st.metric("Last Close", f"${last_close:.2f}")
try:
    info  = yf.Ticker(ticker).info
    live  = info.get("regularMarketPrice", last_close)
    prev  = info.get("previousClose",    last_close)
    pct   = (live - prev) / prev * 100
    arrow = "🔺" if pct >= 0 else "🔻"
    color = "green" if pct >= 0 else "red"
    st.markdown(f"""
    <div style="padding:10px;background:#f0f2f6;border-radius:8px;display:flex;justify-content:space-between;">
      <span style="font-size:24px;font-weight:bold">{ticker}</span>
      <span style="font-size:20px;color:{color};">${live:.2f} {arrow} {pct:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"⚠️ Live ticker unavailable: {e}")

# --- Earnings & Dividend Info ---
try:
    tkr = yf.Ticker(ticker)
    cal = tkr.calendar

    # Next earnings date (handle both DataFrame and dict)
    if isinstance(cal, dict):
        edates = cal.get('Earnings Date')
        if edates:
            nxt = edates[0] if isinstance(edates, (list, tuple)) else edates
            nxt = pd.to_datetime(nxt).date()
            st.markdown(f"**🗓️ Next Earnings Date:** {nxt}")
        else:
            st.markdown("**🗓️ Next Earnings Date:** N/A")
    else:
        if 'Earnings Date' in cal.index:
            nxt = cal.loc['Earnings Date'][0]
            nxt = pd.to_datetime(nxt).date()
            st.markdown(f"**🗓️ Next Earnings Date:** {nxt}")
        else:
            st.markdown("**🗓️ Next Earnings Date:** N/A")

    # Last quarterly earnings & EPS
    qearn = tkr.quarterly_earnings
    if isinstance(qearn, pd.DataFrame) and not qearn.empty:
        last_date = pd.to_datetime(qearn.index[-1]).date()
        last_eps  = qearn['Earnings'].iloc[-1]
        st.markdown(f"**📊 Last Earnings (Quarter):** {last_date} (EPS: ${last_eps:.2f})")
    else:
        st.markdown("**📊 Last Earnings (Quarter):** N/A")

    # Last dividend
    divs = tkr.dividends
    if isinstance(divs, (pd.Series, pd.DataFrame)) and not divs.empty:
        # if DataFrame, take a column; if Series, use directly
        if isinstance(divs, pd.DataFrame):
            divs = divs.iloc[:, 0]
        div_date = divs.index[-1].date()
        div_amt  = float(divs.iloc[-1])
        st.markdown(f"**💰 Last Dividend:** {div_date} (${div_amt:.2f})")
    else:
        st.markdown("**💰 Dividend:** N/A")

except Exception as e:
    st.warning(f"⚠️ Could not fetch earnings/dividend info: {e}")


# --- Feature Engineering ---
delta = df['Close'].diff()
gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
avg_g = gain.ewm(span=14, adjust=False).mean()
avg_l = loss.ewm(span=14, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + avg_g/avg_l))
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()
tr = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low']  - df['Close'].shift()).abs()
], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14).mean()
df['Vol_Spike'] = (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5).astype(int)
try:
    ed = pd.to_datetime(yf.Ticker(ticker).calendar.loc['Earnings Date'][0]).date()
    df['Earnings_Flag'] = (df.index.date == ed).astype(int)
except:
    df['Earnings_Flag'] = 0
df.dropna(inplace=True)

# --- Plot Price & RSI ---
st.subheader("📈 Price & RSI")
st.line_chart(df[['Close','RSI']])

# --- Prepare modeling frame ---
mdl = pd.DataFrame({
    'ds': df.index,
    'y': df['Close'],
    'RSI': df['RSI'],
    'MACD': df['MACD'],
    'ATR': df['ATR'],
    'Vol_Spike': df['Vol_Spike'],
    'Earnings_Flag': df['Earnings_Flag']
})
mdl['Lag1'] = mdl['y'].shift(1)
mdl['Lag2'] = mdl['y'].shift(2)
mdl.dropna(inplace=True)

# --- Train/test split ---
split = int(len(mdl) * 0.8)
train, test = mdl.iloc[:split], mdl.iloc[split:]
features = ['Lag1','Lag2','RSI','MACD','ATR','Vol_Spike','Earnings_Flag']
X_train, y_train = train[features], train['y'].values.ravel()
X_test,  y_test  = test[features],  test['y'].values.ravel()

# --- Models & hyperparameter tuning ---
st.sidebar.markdown("### ✅ Models & Tuning")
use_rf   = st.sidebar.checkbox("Random Forest",     True)
use_lr   = st.sidebar.checkbox("Linear Regression", True)
use_xgb  = st.sidebar.checkbox("XGBoost",           True)
use_prop = st.sidebar.checkbox("Prophet",           True)
use_ari  = st.sidebar.checkbox("ARIMA",             True)
use_lstm = st.sidebar.checkbox("LSTM",              True)
tune_hp  = st.sidebar.checkbox("Tune RF/XGB HP",    False)

results = {}

# Random Forest
if use_rf:
    if tune_hp:
        params = {'n_estimators':[50,100,200],'max_depth':[None,5,10],'min_samples_split':[2,5]}
        model = RandomizedSearchCV(RandomForestRegressor(), params, n_iter=5, cv=3).fit(X_train, y_train).best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, pred)
    fc   = model.predict(mdl[features].iloc[-horizon:])
    results["Random Forest"] = {"pred": pred[-1], "mape": mape, "forecast": fc}

# Linear Regression
if use_lr:
    model = LinearRegression().fit(X_train, y_train)
    pred  = model.predict(X_test)
    mape  = mean_absolute_percentage_error(y_test, pred)
    fc    = model.predict(mdl[features].iloc[-horizon:])
    results["Linear Regression"] = {"pred": pred[-1], "mape": mape, "forecast": fc}

# XGBoost
if use_xgb:
    if tune_hp:
        params = {'n_estimators':[50,100,200],'learning_rate':[0.01,0.1,0.2],'max_depth':[3,5,7]}
        model = RandomizedSearchCV(XGBRegressor(objective='reg:squarederror'), params, n_iter=5, cv=3).fit(X_train, y_train).best_estimator_
    else:
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X_train, y_train)
    pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, pred)
    fc   = model.predict(mdl[features].iloc[-horizon:])
    results["XGBoost"] = {"pred": pred[-1], "mape": mape, "forecast": fc}

# Prophet
if use_prop:
    try:
        pdf = mdl[['ds','y']].copy()
        pdf['y'] = pdf['y'].astype(float)
        m   = Prophet().fit(pdf)
        fut = m.make_future_dataframe(periods=horizon)
        pr  = m.predict(fut)['yhat'].iloc[-horizon:].values
        mape = mean_absolute_percentage_error(mdl['y'].iloc[-horizon:].values, pr)
        results["Prophet"] = {"pred": pr[-1], "mape": mape, "forecast": pr}
    except Exception as e:
        st.warning(f"Prophet skipped: {e}")

# ARIMA (AIC grid search)
if use_ari:
    best_aic, best_order = np.inf, (1,1,0)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    mA = ARIMA(df['Close'], order=(p,d,q)).fit()
                    if mA.aic < best_aic:
                        best_aic, best_order = mA.aic, (p,d,q)
                except:
                    pass
    mA   = ARIMA(df['Close'], order=best_order).fit()
    pr   = mA.forecast(steps=horizon).values
    mape = mean_absolute_percentage_error(df['Close'].iloc[-horizon:].values, pr)
    results["ARIMA"] = {"pred": pr[-1], "mape": mape, "forecast": pr}

# LSTM
if use_lstm:
    scaler = MinMaxScaler().fit(df['Close'].values.reshape(-1,1))
    scaled = scaler.transform(df['Close'].values.reshape(-1,1))
    Xl, Yl = [], []
    for i in range(60, len(scaled)-horizon):
        Xl.append(scaled[i-60:i,0])
        Yl.append(scaled[i:i+horizon,0])
    Xl, Yl = np.array(Xl), np.array(Yl)
    Xl = Xl.reshape((Xl.shape[0], 60, 1))
    lstm = Sequential([LSTM(50, return_sequences=True, input_shape=(60,1)),
                       LSTM(50),
                       Dense(horizon)])
    lstm.compile('adam','mse')
    lstm.fit(Xl, Yl, epochs=5, batch_size=16, verbose=0)
    pr   = lstm.predict(Xl[-1].reshape(1,60,1))[0]
    pred = scaler.inverse_transform(pr.reshape(-1,1)).flatten()
    mape = mean_absolute_percentage_error(df['Close'].iloc[-horizon:].values, pred)
    results["LSTM"] = {"pred": pred[-1], "mape": mape, "forecast": pred}

# Ensemble (inverse-MAPE weights)
weights = np.array([1/v['mape'] for v in results.values() if v['mape'] is not None])
weights /= weights.sum()
ens_fc  = sum(w * np.array(v['forecast']) for w, v in zip(weights, [v for v in results.values() if v['mape'] is not None]))
results["Ensemble"] = {"pred": ens_fc[-1], "mape": None, "forecast": ens_fc}

# --- Recommendation ---
valid_models = {k:v for k,v in results.items() if v['mape'] is not None}
best_model   = min(valid_models, key=lambda k: valid_models[k]['mape'])
recommend    = "Buy ✅" if valid_models[best_model]['forecast'][-1] > last_close else "Hold ⚖️"
st.subheader("📈 Recommendation")
st.success(f"Based on {best_model} (lowest MAPE), recommendation: **{recommend}**")

# --- MAPE Explanation ---
st.caption("📘 MAPE = Mean Absolute Percentage Error, the average absolute percent difference between predictions and actuals (lower is better).")

# --- Display metrics & forecasts ---
st.subheader("📊 Model Predictions & Metrics")
for name, d in results.items():
    m_text = f"{d['mape']:.2%}" if d.get('mape') is not None else "N/A"
    st.write(f"**{name}** → Pred: ${d['pred']:.2f} | MAPE: {m_text}")

st.subheader(f"📅 {horizon}-Day Forecast Comparison")
fc_df = pd.DataFrame({'Date': pd.date_range(df.index[-1] + timedelta(days=1), periods=horizon)})
for name in results:
    fc_df[name] = results[name]['forecast']
st.dataframe(fc_df)

st.subheader("📉 Model Confidence (Inverse-MAPE Weights)")
fig = go.Figure(go.Bar(
    x=list(results.keys()),
    y=list(weights),
    text=[f"{w:.2%}" for w in weights],
    textposition="auto"
))
fig.update_layout(xaxis_title="Model", yaxis_title="Weight")
st.plotly_chart(fig, use_container_width=True)
