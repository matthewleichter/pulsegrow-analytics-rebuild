# timeseries_forecast.py

import pandas as pd
import numpy as np
import os

from utils.timesnet_utils import forecast_usage_with_xgboost, plot_usage_forecast
#from utils.timesnet_utils import TimesNetPredictor
from utils.preprocessing import preprocess_timeseries
from utils.visualization import plot_forecast

DATA_PATH = "data/usage_logs.csv"
OUTPUT_PREDICTIONS = "outputs/timesnet_forecast.csv"
PLOT_PATH = "outputs/timesnet_forecast_plot.png"

def forecast_timeseries():
    print("🔮 Starting time series forecasting with XGBoost...")

    # Load and preprocess the time series data
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)
    ts_data = preprocess_timeseries(df)

    # Define the forecasting model
    print("🧠 Initializing TimesNet model...")
    model = forecast_usage_with_xgboost(df, input_window=30, forecast_horizon=5)

    # Fit the model
    print("🎯 Training model...")
    model.fit(ts_data)

    # Forecast the next steps
    print("📈 Forecasting future values...")
    forecast = model.predict(ts_data)

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
    plot_forecast(ts_data, forecast, save_path=PLOT_PATH)
    print(f"📷 Plot saved to {PLOT_PATH}")

    print("✅ Time series forecast completed.")

if __name__ == "__main__":
    forecast_timeseries()
