import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_usage(df):
    model = ExponentialSmoothing(df["usage"], trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(5)
    return forecast