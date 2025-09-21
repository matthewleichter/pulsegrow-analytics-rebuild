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