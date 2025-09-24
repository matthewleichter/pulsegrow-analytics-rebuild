import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def perform_segmentation(data, n_clusters=3):
    features = data.select_dtypes(include=['float64', 'int64']).copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['segment'] = kmeans.fit_predict(scaled_features)

    return data, kmeans

def plot_segment_clusters(data):
    if 'segment' not in data.columns:
        raise ValueError("Data must have a 'segment' column")

    plt.figure(figsize=(8, 6))
    for segment in data['segment'].unique():
        segment_data = data[data['segment'] == segment]
        plt.scatter(segment_data.iloc[:, 0], segment_data.iloc[:, 1], label=f"Segment {segment}")

    plt.title("Customer Segmentation Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    return plt.gcf()

def load_segmentation_results(path="data/segmentation_results.csv"):
    """
    Loads precomputed segmentation results (e.g., from KMeans or GMM).
    """
    return pd.read_csv(path)

def load_segmentation_data():
    return pd.read_csv('data/segmentation_labels.csv')

def plot_segments(df):
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(df[['feature1', 'feature2']])
    plt.scatter(df['feature1'], df['feature2'], c=clusters)
    plt.title("User Segments")
    return plt.gcf()
