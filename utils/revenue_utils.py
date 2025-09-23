import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def load_revenue_data():
    return pd.read_csv('data/marketing_spend.csv')

def preprocess_revenue_data(transactions_df, marketing_df):
    """
    Merges transaction and marketing spend data, aggregates daily revenue,
    and returns a cleaned dataframe suitable for forecasting.
    """
    if 'date' not in transactions_df.columns or 'revenue' not in transactions_df.columns:
        raise ValueError("transactions.csv must have 'date' and 'revenue' columns")

    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    marketing_df['date'] = pd.to_datetime(marketing_df['date'])

    # Aggregate transactions to daily revenue
    daily_revenue = transactions_df.groupby('date')['revenue'].sum().reset_index()

    # Optional: Merge with marketing data (e.g., to model effects later)
    merged = pd.merge(daily_revenue, marketing_df, on='date', how='left')

    return merged

def forecast_revenue(df):
    """
    Fits a Prophet model to the revenue time series and returns the forecast DataFrame.
    """
    # Prophet expects 'ds' and 'y' column names
    prophet_df = df[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=14)  # Forecast 14 days ahead
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def plot_revenue_forecast(df):
    df['rolling'] = df['revenue'].rolling(3).mean()
    plt.plot(df['revenue'], label='Revenue')
    plt.plot(df['rolling'], label='Rolling Mean')
    plt.legend()
    plt.title("Revenue Forecast")
    return plt.gcf()
