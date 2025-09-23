import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution(df, column, title='Distribution'):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()

def plot_correlation_matrix(df, title='Correlation Matrix'):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.title(title)
    plt.show()

def plot_forecast(ts_data, forecast, save_path=None):
    """
    Plots historical usage data and forecasted values.

    Parameters:
    - ts_data: pandas Series or DataFrame with datetime index and single 'usage' column
    - forecast: list or array of forecasted values
    - save_path: if provided, saves the plot to this file path
    """
    plt.figure(figsize=(14, 6))
    plt.plot(ts_data.index, ts_data.values, label='Historical Usage', linewidth=2)
    
    future_index = pd.date_range(start=ts_data.index[-1], periods=len(forecast)+1, freq="D")[1:]
    plt.plot(future_index, forecast, label='Forecasted Usage', linestyle='--', marker='o')

    plt.title("Usage Forecast (XGBoost)")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
