import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_forecast_data():
    return pd.read_csv('data/usage_logs.csv')

def forecast_next_points(df, steps=5):
    model = ARIMA(df['usage_count'], order=(3,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    plt.plot(df['usage_count'], label='History')
    plt.plot(range(len(df), len(df)+steps), forecast, label='Forecast')
    plt.title("Usage Forecast")
    plt.legend()
    return plt.gcf()