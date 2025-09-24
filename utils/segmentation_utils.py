import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
