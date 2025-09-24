import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_segmentation(df, num_clusters=4):
    # Keep only numeric features for clustering
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns found for clustering.")

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

    # Attach cluster labels to original dataframe
    clustered_df = df.copy()
    clustered_df["Cluster"] = labels

    return clustered_df, labels


def load_segmentation_results(clustered_df):
    if "Cluster" not in clustered_df.columns:
        raise ValueError("No 'Cluster' column found in DataFrame. Run perform_segmentation first.")
    
    labels = clustered_df["Cluster"].values
    return clustered_df, labels


def plot_segmentation(df, labels):
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["Cluster"], errors="ignore")

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for plotting clusters.")

    x = numeric_df.iloc[:, 0]
    y = numeric_df.iloc[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.title("Customer Segments (First 2 Features)")
    plt.grid(True)
    return plt.gcf()
