import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# ——— Streamlit config ———
st.set_page_config(page_title="📈 Stock & Crypto Predictor", layout="wide")
st.title("🔮 Real-Time Stock & Crypto Dashboard")
st.write("🔄 Auto-refreshing every 60 seconds…")

# ——— Sidebar settings ———
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

# ——— Fetch & clean price data ———
raw = yf.download(ticker, start=start_date, end=end_date, group_by=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(0)
df = raw[['Open','High','Low','Close','Volume']].copy()
if df.empty:
    st.error("❌ No data—check your ticker")
    st.stop()

# ——— Enforce business days & fill gaps ———
df.index = pd.DatetimeIndex(df.index)
df = df.asfreq('B')
df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].ffill()
df.dropna(inplace=True)

# ——— Live price display ———
last_close = float(df['Close'].iloc[-1])
st.subheader(f"💰 Current Price for {ticker}")
st.metric("Last Close", f"${last_close:.2f}")
try:
    info = yf.Ticker(ticker).info
    live = info.get("regularMarketPrice", last_close)
    prev = info.get("previousClose",    last_close)
    pct  = (live - prev) / prev * 100
    arrow = "🔺" if pct>=0 else "🔻"
    color = "green" if pct>=0 else "red"
    st.markdown(f"""
      <div style="padding:10px;background:#f0f2f6;border-radius:8px;
                  display:flex;justify-content:space-between;">
        <span style="font-size:24px;font-weight:bold">{ticker}</span>
        <span style="font-size:20px;color:{color};">
          ${live:.2f} {arrow} {pct:.2f}%
        </span>
      </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"⚠️ Live data unavailable: {e}")

# ——— Earnings & Dividend Info ———
try:
    tkr = yf.Ticker(ticker)
    cal = tkr.calendar
    if isinstance(cal, dict):
        ed = cal.get('Earnings Date')
        nxt = pd.to_datetime(ed[0] if isinstance(ed,(list,tuple)) else ed).date() if ed else None
    else:
        nxt = pd.to_datetime(cal.loc['Earnings Date'][0]).date() if 'Earnings Date' in cal.index else None
    st.markdown(f"**🗓️ Next Earnings Date:** {nxt if nxt else 'N/A'}")

    qearn = tkr.quarterly_earnings
    if isinstance(qearn, pd.DataFrame) and not qearn.empty:
        ld  = pd.to_datetime(qearn.index[-1]).date()
        eps = qearn['Earnings'].iloc[-1]
        st.markdown(f"**📊 Last Earnings (Quarter):** {ld} (EPS: ${eps:.2f})")
    else:
        st.markdown("**📊 Last Earnings (Quarter):** N/A")

    divs = tkr.dividends
    if len(divs):
        divs = divs.sort_index()
        if isinstance(divs, pd.DataFrame): divs = divs.iloc[:,0]
        dd = divs.index[-1].date(); da = float(divs.iloc[-1])
        st.markdown(f"**💰 Last Dividend:** {dd} (${da:.2f})")
    else:
        st.markdown("**💰 Last Dividend:** N/A")

    ex_ts = tkr.info.get("exDividendDate")
    ex_date = datetime.fromtimestamp(ex_ts).date() if ex_ts else None
    st.markdown(f"**🪙 Ex-Dividend Date:** {ex_date if ex_date else 'N/A'}")

except Exception as e:
    st.warning(f"⚠️ Could not fetch earnings/dividend info: {e}")

# ——— Market Breadth & Macro series ———
macros = ["^VIX","^TNX","SPY","XLK","XLF"]
st.subheader("📈 Market Breadth & Macro")
try:
    rawm = yf.download(macros, start=start_date, end=end_date, group_by="ticker")
    macro_df = pd.DataFrame({
        m: rawm[m]['Close'] if (m in rawm and 'Close' in rawm[m])
           else rawm[m].droplevel(0,axis=1)['Close']
        for m in macros
    })
    macro_df = macro_df.asfreq('B').ffill().loc[df.index]
    latest = macro_df.iloc[-1]
    st.write(f"VIX: {latest['^VIX']:.2f}   |   TNX: {latest['^TNX']:.2f}%")
    st.write(f"SPY: {latest['SPY']:.2f}   |   XLK: {latest['XLK']:.2f}   |   XLF: {latest['XLF']:.2f}")
except Exception as e:
    st.error(f"Failed to load breadth/macro data: {e}")

# ——— Correlation vs. Stock Price ———
st.subheader("🔗 Correlation vs. Stock Price")
macro_ret = macro_df.pct_change().add_suffix("_ret").dropna()
corr_df   = pd.concat([df['Close'], macro_ret], axis=1).dropna()
corrs     = corr_df.corr().loc['Close', macro_ret.columns]
colors    = ['green' if v>=0 else 'red' for v in corrs]
fig_corr = go.Figure(go.Bar(
    x=corrs.index, y=corrs.values,
    marker_color=colors,
    text=[f"{v:.2f}" for v in corrs.values],
    textposition="auto"
))
fig_corr.update_layout(yaxis_title="Correlation with Stock Close")
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown("""
**Legend**  
- 🟢 **Positive**: Macro moves *with* the stock price  
- 🔴 **Negative**: Macro moves *against* the stock price
""")

# ——— Technical Indicators ———
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
    ed = pd.to_datetime(tkr.calendar.loc['Earnings Date'][0]).date()
    df['Earnings_Flag'] = (df.index.date==ed).astype(int)
except:
    df['Earnings_Flag'] = 0

st.subheader("📈 Price & RSI")
st.line_chart(df[['Close','RSI']])

# ——— Modeling ———
@st.cache_data(ttl=600, show_spinner=False)
def train_models(df, macro_ret, horizon):
    feats = ['Lag1','Lag2','RSI','MACD','ATR','Vol_Spike','Earnings_Flag'] + list(macro_ret.columns)
    mdl = pd.DataFrame({'ds': df.index, 'y': df['Close']})
    for f in feats:
        if f in macro_ret:
            mdl[f] = macro_ret[f]
        elif f=='Lag1':
            mdl[f] = df['Close'].shift(1)
        elif f=='Lag2':
            mdl[f] = df['Close'].shift(2)
        else:
            mdl[f] = df[f]
    mdl.dropna(inplace=True)

    split = int(len(mdl)*0.8)
    train, test = mdl.iloc[:split], mdl.iloc[split:]
    Xtr, ytr = train[feats], train['y']
    Xte, yte = test[feats],  test['y']

    results = {}
    test_preds = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100).fit(Xtr,ytr)
    pr_rf = rf.predict(Xte)
    results["Random Forest"] = {
      "pred": pr_rf[-1],
      "mape": mean_absolute_percentage_error(yte,pr_rf),
      "forecast": rf.predict(mdl[feats].iloc[-horizon:])
    }
    test_preds["Random Forest"] = pr_rf

    # XGBoost
    xgb = XGBRegressor(objective='reg:squarederror',n_estimators=100).fit(Xtr,ytr)
    pr_xg = xgb.predict(Xte)
    results["XGBoost"] = {
      "pred": pr_xg[-1],
      "mape": mean_absolute_percentage_error(yte,pr_xg),
      "forecast": xgb.predict(mdl[feats].iloc[-horizon:])
    }
    test_preds["XGBoost"] = pr_xg

    # Prophet
    pdf = pd.DataFrame({'ds':train.index,'y':train['y'].astype(float)})
    pm  = Prophet().fit(pdf)
    df_test = pd.DataFrame({'ds':test.index})
    pr_pt = pm.predict(df_test)['yhat'].values
    results["Prophet"] = {
      "pred": pr_pt[-1],
      "mape": mean_absolute_percentage_error(yte, pr_pt),
      "forecast": pm.predict(pm.make_future_dataframe(periods=horizon))['yhat'].iloc[-horizon:].values
    }
    test_preds["Prophet"] = pr_pt

    # ARIMA
    best_aic, best_ord = np.inf,(1,1,0)
    for p in range(3):
      for d in range(2):
        for q in range(3):
          try:
            mA = SARIMAX(df['Close'],
                         order=(p,d,q), trend='ct',
                         enforce_stationarity=False,
                         enforce_invertibility=False
                       ).fit(disp=False)
            if mA.aic<best_aic:
              best_aic,best_ord=mA.aic,(p,d,q)
          except: pass
    arma = SARIMAX(df['Close'],
                   order=best_ord, trend='ct',
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)
    pr_ar = arma.predict(start=test.index[0], end=test.index[-1]).values
    frc_ar = arma.forecast(steps=horizon).values
    results["ARIMA"] = {
      "pred": float(frc_ar[-1]),
      "mape": mean_absolute_percentage_error(yte,pr_ar),
      "forecast": frc_ar
    }
    test_preds["ARIMA"] = pr_ar

    # LSTM
    scaler = MinMaxScaler().fit(df['Close'].values.reshape(-1,1))
    scaled = scaler.transform(df['Close'].values.reshape(-1,1))
    Xl,Yl = [],[]
    for i in range(60,len(scaled)-horizon):
      Xl.append(scaled[i-60:i,0]); Yl.append(scaled[i:i+horizon,0])
    Xl = np.array(Xl).reshape(-1,60,1); Yl = np.array(Yl)
    lstm = Sequential([LSTM(50,return_sequences=True,input_shape=(60,1)),
                      LSTM(50),Dense(horizon)])
    lstm.compile('adam','mse')
    lstm.fit(Xl,Yl,epochs=5,batch_size=16,verbose=0)
    pr_ls = []
    for dt in test.index:
      pos = df.index.get_loc(dt)
      seq = scaled[pos-60:pos,0].reshape(1,60,1)
      p1 = lstm.predict(seq,verbose=0)[0][0]
      pr_ls.append(float(scaler.inverse_transform([[p1]])[0,0]))
    frc_ls = scaler.inverse_transform(
                lstm.predict(scaled[-60:].reshape(1,60,1),verbose=0)
                  .reshape(-1,1)).flatten()
    results["LSTM"] = {
      "pred": float(frc_ls[-1]),
      "mape": mean_absolute_percentage_error(yte,pr_ls),
      "forecast": frc_ls
    }
    test_preds["LSTM"] = np.array(pr_ls)

    return results, test.index, test_preds

# ——— Train & Forecast ———
if st.sidebar.button("▶️ Train & Forecast"):
    with st.spinner("Training all models…"):
        results, test_idx, test_preds = train_models(df, macro_ret, horizon)

    # Recommendation
    valid = {k:v for k,v in results.items() if v['mape'] is not None}
    best  = min(valid, key=lambda k: valid[k]['mape'])
    rec   = "Buy ✅" if valid[best]['forecast'][-1] > last_close else "Hold ⚖️"
    st.subheader("📈 Recommendation")
    st.success(f"Based on {best} (lowest MAPE), recommendation: **{rec}**")
    st.caption("📘 MAPE = Mean Absolute Percentage Error (lower is better).")

    # Model metrics
    st.subheader("📊 Model Predictions & Metrics")
    for name,d in results.items():
        st.write(f"**{name}** → Pred: ${d['pred']:.2f} | MAPE: {d['mape']:.2%}")

    # Forecast table
    st.subheader(f"📅 {horizon}-Day Forecast Comparison")
    fc = pd.DataFrame({m:results[m]['forecast'] for m in results},
                      index=pd.date_range(df.index[-1]+timedelta(days=1),
                                          periods=horizon, freq='B'))
    st.dataframe(fc.round(2))

    # 7-Day Backtest Accuracy table
    back_h = 7
    idx    = test_idx[-back_h:]
    back_df = pd.DataFrame({m: test_preds[m][-back_h:] for m in test_preds}, index=idx)
    back_df['Actual Close'] = df['Close'].loc[idx]
    st.subheader("📉 7-Day Backtest: Actual vs Predicted")
    st.dataframe(back_df.round(2))

else:
    st.info("▶️ Click **Train & Forecast** in the sidebar to run models and view recommendations.")
