#dependencies are preprocessing and timesnet_utils in the utils folder and timesnet in the models folder

# scripts/timeseries_forecasting.py

import pandas as pd
import numpy as np
import os

from utils.timesnet_utils import forecast_usage_with_xgboost, plot_usage_forecast
from utils.preprocessing import preprocess_timeseries

DATA_PATH = "data/usage_logs.csv"
OUTPUT_PREDICTIONS = "outputs/timesnet_forecast.csv"
PLOT_PATH = "outputs/timesnet_forecast_plot.png"

def forecast_timeseries():
    print("🔮 Starting time series forecasting with XGBoost...")

    # Load and preprocess the time series data
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)
    ts_data = preprocess_timeseries(df)

    # Extract just the numeric usage array from preprocessed data
    usage_array = ts_data["usage_normalized"].values

    # Forecast using utility function
    print("📈 Forecasting future values...")
    forecast = forecast_usage_with_xgboost(ts_data, input_window=30, forecast_horizon=5)

    # Save forecast
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    forecast_df = pd.DataFrame({
        "timestamp": pd.date_range(start=ts_data.index[-1], periods=len(forecast)+1, freq="D")[1:],
        "forecast": forecast
    })
    forecast_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"💾 Forecast saved to {OUTPUT_PREDICTIONS}")

    # Plot the forecast
    print("📊 Plotting forecast...")
    plot = plot_usage_forecast(ts_data, forecast)
    plot.savefig(PLOT_PATH)
    print(f"📷 Plot saved to {PLOT_PATH}")

    print("✅ Time series forecast completed.")

if __name__ == "__main__":
    forecast_timeseries()
