# utils/timesnet_utils.py (now using XGBoost instead of TimesNet)

import pandas as pd
import matplotlib.pyplot as plt
from models.timesnet_predictor import XGBoostPredictor


def forecast_usage_with_xgboost(df, input_window=30, forecast_horizon=5):
    """
    Uses XGBoost to forecast the next `forecast_horizon` usage points
    based on the past `input_window` values in the dataframe.
    """
    if "usage" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'usage' column.")

    usage_series = df["usage"].dropna().values[-input_window:]

    model = XGBoostPredictor(input_window=input_window, forecast_horizon=forecast_horizon)
    forecast = model.predict(usage_series)

    return forecast


def plot_usage_forecast(df, forecast, forecast_horizon=5):
    """
    Plots the original usage data and appends the forecast to the end.
    """
    plt.figure(figsize=(10, 5))
    past = df["usage"].dropna().values[-30:]  # Plot last 30 points
    forecast_range = list(range(len(past), len(past) + forecast_horizon))

    plt.plot(range(len(past)), past, label="Past Usage", marker='o')
    plt.plot(forecast_range, forecast, label="Forecast", linestyle="--", marker='x')
    plt.xlabel("Time")
    plt.ylabel("Usage")
    plt.title("Usage Forecast with XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    return plt
