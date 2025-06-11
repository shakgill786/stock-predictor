import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def prepare_lstm_data(series, window_size=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_lstm_model(series, window_size=10, epochs=10):
    X, y, scaler = prepare_lstm_data(series, window_size)
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    return model, scaler, X, y

def predict_lstm(model, scaler, series, window_size=10, steps=5):
    last_sequence = series.values[-window_size:].reshape(-1, 1)
    scaled_seq = scaler.transform(last_sequence)
    pred_input = scaled_seq.reshape((1, window_size, 1))
    predictions = []

    for _ in range(steps):
        pred = model.predict(pred_input, verbose=0)[0][0]
        predictions.append(pred)
        pred_input = np.append(pred_input[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
