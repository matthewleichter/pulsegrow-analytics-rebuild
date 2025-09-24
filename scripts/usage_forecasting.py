import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

def run_usage_forecasting():
    st.title("Usage Forecasting with Smoothing and Confidence Intervals")

    uploaded_file = st.file_uploader("Upload usage data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.set_index("date", inplace=True)

        st.write("Uploaded Data", df.head())

        # Smoothing with rolling average
        df["smoothed"] = df["usage"].rolling(window=5, min_periods=1).mean()

        # Basic Forecast: naive continuation
        forecast_horizon = 10
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

        mean_usage = df["smoothed"].iloc[-5:].mean()
        std_usage = df["smoothed"].iloc[-5:].std()

        forecast_values = np.full(forecast_horizon, mean_usage)
        lower_bound = forecast_values - 1.96 * std_usage
        upper_bound = forecast_values + 1.96 * std_usage

        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast_values,
            "lower": lower_bound,
            "upper": upper_bound
        }).set_index("date")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 4))
        df["usage"].plot(ax=ax, label="Actual Usage", alpha=0.6)
        df["smoothed"].plot(ax=ax, label="Smoothed", linewidth=2)
        forecast_df["forecast"].plot(ax=ax, label="Forecast", style="--")
        ax.fill_between(forecast_df.index, forecast_df["lower"], forecast_df["upper"], color='gray', alpha=0.3, label="95% CI")
        ax.set_title("Usage Forecast with Confidence Intervals")
        ax.set_ylabel("Usage")
        ax.legend()
        st.pyplot(fig)
