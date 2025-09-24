import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_usage_forecast(df):
    fig, ax = plt.subplots()
    df.plot(x='date', y='usage', ax=ax, label='Raw Usage')
    if 'smoothed' in df.columns:
        df.plot(x='date', y='smoothed', ax=ax, label='Smoothed', linestyle='--')
    ax.set_title("Usage Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Usage")
    ax.legend()
    return fig

def plot_churn_results(df):
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='user_segment', y='churn_probability', ax=ax)
    ax.set_title("Churn Prediction by Segment")
    ax.set_xlabel("User Segment")
    ax.set_ylabel("Churn Probability")
    return fig

def plot_segmentation_clusters(df):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['x'], df['y'], c=df['cluster'], cmap='tab10', alpha=0.6)
    ax.set_title("User Segmentation Clusters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    return fig

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

def plot_forecast_with_confidence(forecast_df):
    """
    Plots Prophet forecast with confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast")
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                     alpha=0.3, label="Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.title("Revenue Forecast with Prophet")
    return plt.gcf()
