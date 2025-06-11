import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def train_arima_model(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def predict_arima(model_fit, steps=5):
    forecast = model_fit.forecast(steps=steps)
    return forecast.values if hasattr(forecast, 'values') else forecast

def evaluate_arima(model_fit, actual):
    pred = model_fit.predict(start=0, end=len(actual)-1)
    return np.sqrt(mean_squared_error(actual, pred))
