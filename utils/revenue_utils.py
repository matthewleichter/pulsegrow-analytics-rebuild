import pandas as pd
import matplotlib.pyplot as plt

def load_revenue_data():
    return pd.read_csv('data/marketing_spend.csv')

def plot_revenue_forecast(df):
    df['rolling'] = df['revenue'].rolling(3).mean()
    plt.plot(df['revenue'], label='Revenue')
    plt.plot(df['rolling'], label='Rolling Mean')
    plt.legend()
    plt.title("Revenue Forecast")
    return plt.gcf()

def preprocess_revenue_data(df):
    """
    Cleans and prepares revenue data for forecasting.
    Expects a 'date' column and a 'revenue' column.
    """
    if 'date' not in df.columns or 'revenue' not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'revenue' columns")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    df = df[['revenue']].dropna()

    return df
