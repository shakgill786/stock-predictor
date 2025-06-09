import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load stock data (Apple as an example)
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Create target column (next day's close price)
df["Tomorrow_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# Features and target
X = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Tomorrow_Close"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")
plt.title(f"{ticker} Price Prediction (Next Day Close)")
plt.legend()
plt.show()
