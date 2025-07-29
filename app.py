import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Page config ---
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

# --- Cached data fetch ---
@st.cache_data(ttl=600, show_spinner=False)
def fetch_data(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, group_by=False)
    # drop any weird multi‐index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

df = fetch_data(ticker, start_date, end_date)
if df.empty:
    st.error("❌ No data—check your ticker")
    st.stop()

# --- Current & live price UI ---
last_close = float(df['Close'].iloc[-1])
st.subheader(f"💰 Current Price for {ticker}")
st.metric("Last Close", f"${last_close:.2f}")

try:
    import yfinance as yf
    info = yf.Ticker(ticker).info
    live = info.get("regularMarketPrice", last_close)
    prev = info.get("previousClose",    last_close)
    pct  = (live - prev) / prev * 100
    arrow = "🔺" if pct>=0 else "🔻"
    color = "green" if pct>=0 else "red"
    st.markdown(f"""
    <div style="padding:10px;background:#f0f2f6;border-radius:8px;display:flex;justify-content:space-between;">
      <span style="font-size:24px;font-weight:bold">{ticker}</span>
      <span style="font-size:20px;color:{color};">${live:.2f} {arrow} {pct:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"⚠️ Live ticker unavailable: {e}")

# --- Earnings & dividend UI ---
try:
    import yfinance as yf
    tkr = yf.Ticker(ticker)
    cal = tkr.calendar

    # Next earnings
    if isinstance(cal, dict):
        ed = cal.get('Earnings Date')
        if ed:
            nxt = ed[0] if isinstance(ed,(list,tuple)) else ed
            nxt = pd.to_datetime(nxt).date()
            st.markdown(f"**🗓️ Next Earnings Date:** {nxt}")
        else:
            st.markdown("**🗓️ Next Earnings Date:** N/A")
    else:
        if 'Earnings Date' in cal.index:
            nxt = pd.to_datetime(cal.loc['Earnings Date'][0]).date()
            st.markdown(f"**🗓️ Next Earnings Date:** {nxt}")
        else:
            st.markdown("**🗓️ Next Earnings Date:** N/A")

    # Last EPS
    qearn = tkr.quarterly_earnings
    if isinstance(qearn, pd.DataFrame) and not qearn.empty:
        ld  = pd.to_datetime(qearn.index[-1]).date()
        eps = qearn['Earnings'].iloc[-1]
        st.markdown(f"**📊 Last Earnings (Quarter):** {ld} (EPS: ${eps:.2f})")
    else:
        st.markdown("**📊 Last Earnings (Quarter):** N/A")

    # Last dividend
    divs = tkr.dividends
    if isinstance(divs, (pd.Series,pd.DataFrame)) and not divs.empty:
        if isinstance(divs, pd.DataFrame): divs = divs.iloc[:,0]
        dd = divs.index[-1].date()
        da = float(divs.iloc[-1])
        st.markdown(f"**💰 Last Dividend:** {dd} (${da:.2f})")
    else:
        st.markdown("**💰 Dividend:** N/A")

except Exception as e:
    st.warning(f"⚠️ Could not fetch earnings/dividend info: {e}")

# --- Feature engineering & Price/RSI plot ---
delta      = df['Close'].diff()
gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
avg_g = gain.ewm(span=14, adjust=False).mean()
avg_l = loss.ewm(span=14, adjust=False).mean()
df['RSI']  = 100 - (100 / (1 + avg_g/avg_l))

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26 - (ema12-ema26).ewm(span=9, adjust=False).mean()

tr = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low']  - df['Close'].shift()).abs()
], axis=1).max(axis=1)
df['ATR']       = tr.rolling(14).mean()
df['Vol_Spike'] = (df['Volume'] > df['Volume'].rolling(10).mean()*1.5).astype(int)

try:
    ed = pd.to_datetime(pd.to_datetime(yf.Ticker(ticker).calendar.loc['Earnings Date'][0])).date()
    df['Earnings_Flag'] = (df.index.date == ed).astype(int)
except:
    df['Earnings_Flag'] = 0

df.dropna(inplace=True)
st.subheader("📈 Price & RSI")
st.line_chart(df[['Close','RSI']])

# --- Model training (lazy imports!) ---
@st.cache_data(ttl=600, show_spinner=False)
def train_models(df, horizon):
    # lazy imports
    from sklearn.ensemble     import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost              import XGBRegressor
    from sklearn.metrics      import mean_absolute_percentage_error
    from prophet              import Prophet
    from statsmodels.tsa.arima.model import ARIMA
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM, Dense
    from sklearn.preprocessing      import MinMaxScaler

    # build modeling frame
    mdl = pd.DataFrame({
        'ds': df.index,
        'y' : df['Close'],
        'RSI': df['RSI'],
        'MACD': df['MACD'],
        'ATR': df['ATR'],
        'Vol_Spike': df['Vol_Spike'],
        'Earnings_Flag': df['Earnings_Flag']
    })
    mdl['Lag1'] = mdl['y'].shift(1)
    mdl['Lag2'] = mdl['y'].shift(2)
    mdl.dropna(inplace=True)

    # train/test split
    split   = int(len(mdl)*0.8)
    train   = mdl.iloc[:split]
    test    = mdl.iloc[split:]
    feats   = ['Lag1','Lag2','RSI','MACD','ATR','Vol_Spike','Earnings_Flag']
    X_train, y_train = train[feats], train['y']
    X_test,  y_test  = test[feats],  test['y']

    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    p  = rf.predict(X_test)
    results["Random Forest"] = {
      "pred": rf.predict(X_train.tail(1))[0],
      "mape": mean_absolute_percentage_error(y_test, p),
      "forecast": rf.predict(mdl[feats].tail(horizon))
    }

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    p  = lr.predict(X_test)
    results["Linear Regression"] = {
      "pred": lr.predict(X_train.tail(1))[0],
      "mape": mean_absolute_percentage_error(y_test, p),
      "forecast": lr.predict(mdl[feats].tail(horizon))
    }

    # XGBoost
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X_train, y_train)
    p   = xgb.predict(X_test)
    results["XGBoost"] = {
      "pred": xgb.predict(X_train.tail(1))[0],
      "mape": mean_absolute_percentage_error(y_test, p),
      "forecast": xgb.predict(mdl[feats].tail(horizon))
    }

    # Prophet
    pdf = mdl[['ds','y']].copy()
    pdf['y'] = pdf['y'].astype(float)
    m   = Prophet().fit(pdf)
    fut = m.make_future_dataframe(periods=horizon)
    pr  = m.predict(fut)['yhat'].iloc[-horizon:].values
    results["Prophet"] = {
      "pred": pr[-1],
      "mape": mean_absolute_percentage_error(mdl['y'].iloc[-horizon:].values, pr),
      "forecast": pr
    }

    # ARIMA (4‐combo grid)
    best_aic, best_order = np.inf, (1,1,0)
    for p in (0,1):
      for d in (1,):
        for q in (0,1):
          try:
            mA = ARIMA(df['Close'], order=(p,d,q)).fit()
            if mA.aic < best_aic:
                best_aic, best_order = mA.aic, (p,d,q)
          except:
            pass
    mA   = ARIMA(df['Close'], order=best_order).fit()
    arpr = mA.forecast(steps=horizon).values
    results["ARIMA"] = {
      "pred": arpr[-1],
      "mape": mean_absolute_percentage_error(df['Close'].iloc[-horizon:].values, arpr),
      "forecast": arpr
    }

    # LSTM (1 epoch)
    scaler = MinMaxScaler().fit(df['Close'].values.reshape(-1,1))
    scaled = scaler.transform(df['Close'].values.reshape(-1,1))
    Xl, Yl = [], []
    for i in range(60, len(scaled)-horizon):
      Xl.append(scaled[i-60:i,0])
      Yl.append(scaled[i:i+horizon,0])
    Xl = np.array(Xl).reshape(-1,60,1)
    Yl = np.array(Yl)
    lstm = Sequential([
      LSTM(50, return_sequences=True, input_shape=(60,1)),
      LSTM(50),
      Dense(horizon)
    ])
    lstm.compile('adam','mse')
    lstm.fit(Xl, Yl, epochs=1, batch_size=16, verbose=0)
    pr   = lstm.predict(Xl[-1].reshape(1,60,1))[0]
    pred = scaler.inverse_transform(pr.reshape(-1,1)).flatten()
    results["LSTM"] = {
      "pred": pred[-1],
      "mape": mean_absolute_percentage_error(df['Close'].iloc[-horizon:].values, pred),
      "forecast": pred
    }

    # Ensemble
    weights = np.array([1/v['mape'] for v in results.values()])
    weights /= weights.sum()
    ens_fc   = sum(w * v['forecast'] for w,v in zip(weights, results.values()))
    results["Ensemble"] = {"pred": ens_fc[-1], "mape": None, "forecast": ens_fc}

    return results, weights

# --- Train & Forecast button ---
if st.sidebar.button("▶️ Train & Forecast"):
    with st.spinner("Training models & generating forecasts…"):
        results, weights = train_models(df, horizon)

    # Recommendation
    valid = {k:v for k,v in results.items() if v['mape'] is not None}
    best  = min(valid, key=lambda k: valid[k]['mape'])
    rec   = "Buy ✅" if valid[best]['forecast'][-1] > last_close else "Hold ⚖️"

    st.subheader("📈 Recommendation")
    st.success(f"Based on {best} (lowest MAPE), recommendation: **{rec}**")
    st.caption("📘 MAPE = Mean Absolute Percentage Error (lower is better).")

    # Metrics & forecasts
    st.subheader("📊 Model Predictions & Metrics")
    for name,d in results.items():
        m_txt = f"{d['mape']:.2%}" if d['mape'] is not None else "N/A"
        st.write(f"**{name}** → Pred: ${d['pred']:.2f} | MAPE: {m_txt}")

    # 7-Day Forecast table
    st.subheader(f"📅 {horizon}-Day Forecast Comparison")
    fc_df = pd.DataFrame({'Date': pd.date_range(df.index[-1]+timedelta(days=1), periods=horizon)})
    for name,d in results.items():
        fc_df[name] = d['forecast']
    st.dataframe(fc_df)

    # Confidence bar chart (lazy Plotly import)
    from plotly import graph_objects as go
    st.subheader("📉 Model Confidence (Inverse-MAPE Weights)")
    fig = go.Figure(go.Bar(
        x=list(results.keys()),
        y=weights,
        text=[f"{w:.2%}" for w in weights],
        textposition="auto"
    ))
    fig.update_layout(xaxis_title="Model", yaxis_title="Weight")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("▶️ Click **Train & Forecast** in the sidebar to run models and view recommendations.")
