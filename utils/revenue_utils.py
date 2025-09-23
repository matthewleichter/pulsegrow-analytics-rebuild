# utils/revenue_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def preprocess_revenue_data(transactions_df, marketing_df=None):
    """
    Prepares revenue data for forecasting.
    Expects 'date' and 'revenue' columns in transactions_df.
    """
    df = transactions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').agg({'revenue': 'sum'}).reset_index()
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    df = df.asfreq('D')  # daily frequency (adjust as needed)
    df = df.fillna(method='ffill')
    return df

def forecast_revenue(df, periods=30):
    """
    Forecast future revenue using SARIMAX.
    Returns a DataFrame with historical + forecasted values and confidence intervals.
    """
    model = SARIMAX(df['revenue'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame(alpha=0.05)

    forecast_df['date'] = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
    forecast_df.set_index('date', inplace=True)

    combined = pd.concat([df[['revenue']], forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]], axis=0)
    return combined

def plot_forecast_with_confidence(forecast_df):
    """
    Plots revenue forecast with 95% confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'revenue' in forecast_df.columns:
        forecast_df['revenue'].plot(ax=ax, label='Actual', color='black')

    forecast_df['mean'].plot(ax=ax, label='Forecast', color='blue')
    ax.fill_between(forecast_df.index, 
                    forecast_df['mean_ci_lower'], 
                    forecast_df['mean_ci_upper'], 
                    color='blue', alpha=0.2, label='95% CI')

    ax.set_title("Revenue Forecast with Confidence Interval (ARIMA)")
    ax.legend()
    return fig
