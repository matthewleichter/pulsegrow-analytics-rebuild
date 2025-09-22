import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def load_anomaly_data():
    return pd.read_csv('data/anomaly_logs.csv')

def plot_anomalies(df):
    iso = IsolationForest()
    preds = iso.fit_predict(df[['value']])
    plt.scatter(df.index, df['value'], c=(preds == -1), cmap='coolwarm')
    plt.title("Anomaly Detection")
    return plt.gcf()