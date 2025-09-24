
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from utils.usage_utils import smooth_usage, compute_confidence_interval, forecast_basic

def run_usage_forecasting():
    st.title("Usage Forecasting")

    df = pd.read_csv("data/usage_data.csv")
    st.write("Raw Data", df)

    smoothed_df = smooth_usage(df)
    st.line_chart(smoothed_df[['day', 'smoothed_usage']].set_index('day'))

    lower, upper = compute_confidence_interval(df)
    st.write(f"95% Confidence Interval: [{lower:.2f}, {upper:.2f}]")

    forecast_df = forecast_basic(df)
    st.write("Forecast", forecast_df)

    plt.figure(figsize=(10, 5))
    plt.plot(df['day'], df['usage'], label='Actual')
    plt.plot(smoothed_df['day'], smoothed_df['smoothed_usage'], label='Smoothed')
    plt.plot(forecast_df['day'], forecast_df['forecast'], label='Forecast', linestyle='--')
    plt.fill_between(df['day'], lower, upper, color='gray', alpha=0.2, label='Confidence Interval')
    plt.legend()
    plt.title("Usage Forecast with Smoothing and Confidence Interval")
    plt.xlabel("Day")
    plt.ylabel("Usage")
    plt.grid(True)
    st.pyplot(plt)
