# utils/revenue_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def preprocess_revenue_data(transactions_df, marketing_df):
    """
    Merge and preprocess the revenue and marketing data for forecasting.
    """
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    marketing_df['date'] = pd.to_datetime(marketing_df['date'])

    transactions_df['revenue'] = transactions_df['price'] * transactions_df['quantity']
    revenue_df = transactions_df.groupby('date').agg({'revenue': 'sum'}).reset_index()
    marketing_df = marketing_df.groupby('date').agg({'spend': 'sum'}).reset_index()

    df = pd.merge(revenue_df, marketing_df, on='date', how='outer').fillna(0)
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    return df

def forecast_revenue(df, forecast_days=30, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Forecast revenue using SARIMAX with exogenous variable (marketing spend).
    Returns DataFrame with actuals + forecast + confidence intervals.
    """
    y = df['revenue']
    exog = df[['spend']]

    # Fit SARIMAX model
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    # Prepare future exogenous values (use last known spend as a naive future input)
    future_exog = pd.DataFrame({
        'spend': [exog['spend'].iloc[-1]] * forecast_days
    })

    # Forecast future values
    forecast = model_fit.get_forecast(steps=forecast_days, exog=future_exog)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    forecast_df = pd.DataFrame({
        'forecast': forecast.predicted_mean,
        'lower_ci': forecast.conf_int().iloc[:, 0],
        'upper_ci': forecast.conf_int().iloc[:, 1]
    }, index=forecast_index)

    # Combine historical and forecasted
    full_df = pd.concat([df[['revenue']], forecast_df], axis=0)
    return full_df

def generate_revenue_forecast_plot():
    """
    Load data, forecast using SARIMAX, and return a matplotlib figure.
    """
    transactions = pd.read_csv("data/transactions.csv")
    marketing = pd.read_csv("data/marketing_spend.csv")
    df = preprocess_revenue_data(transactions, marketing)
    forecast_df = forecast_revenue(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    forecast_df['revenue'].plot(ax=ax, label="Actual Revenue", color='blue')
    forecast_df['forecast'].plot(ax=ax, label="Forecasted Revenue", color='orange')
    ax.fill_between(
        forecast_df.index[-30:], 
        forecast_df['lower_ci'].tail(30), 
        forecast_df['upper_ci'].tail(30),
        color='orange', alpha=0.2, label="Confidence Interval"
    )
    ax.set_title("SARIMAX Revenue Forecast with Marketing Spend")
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    return fig
