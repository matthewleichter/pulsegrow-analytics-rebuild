import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load example anomaly data (optional usage for demo/testing)
def load_anomaly_data():
    return pd.read_csv('data/anomaly_logs.csv')

# Main anomaly detection function (called by the app)
def detect_anomalies(df, feature_column="usage", contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(df[[feature_column]])
    return df, model

# Plot anomalies as scatterplot
def plot_anomalies(df, feature_column="usage"):
    if "anomaly" not in df.columns:
        raise ValueError("Missing 'anomaly' column. Run detect_anomalies first.")

    plt.figure(figsize=(10, 4))
    plt.scatter(df.index, df[feature_column], c=(df["anomaly"] == -1), cmap='coolwarm', label="Data")
    plt.title("Anomaly Detection")
    plt.xlabel("Index")
    plt.ylabel(feature_column)
    plt.legend(["Normal", "Anomaly"])
    plt.tight_layout()
    return plt.gcf()
