import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_segmentation(df, num_clusters=4):
    """
    Perform clustering on numeric columns of the DataFrame.
    Returns the clustered DataFrame and cluster labels.
    """
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.empty:
        raise ValueError("No numeric columns found for clustering.")

    # Scale numeric data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # KMeans clustering
    model = KMeans(n_clusters=num_clusters, random_state=42)
    labels = model.fit_predict(scaled_data)

    # Return clustered df with cluster labels
    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    return clustered_df, labels

def plot_segmentation(clustered_df, labels):
    """
    Create a PCA scatter plot colored by cluster label.
    """
    numeric_df = clustered_df.select_dtypes(include='number').drop(columns=['cluster'], errors='ignore')

    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(numeric_df)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    ax.set_title("PCA Projection of Segments")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    return fig

def load_segmentation_results(clustered_df):
    """
    Extracts cluster counts and summary stats from clustered DataFrame.
    Returns a DataFrame with cluster ID and count.
    """
    if "cluster" not in clustered_df.columns:
        raise ValueError("Missing 'cluster' column in clustered_df")

    # Compute basic cluster summary
    cluster_summary = clustered_df.groupby("cluster").size().reset_index(name="Count")
    return cluster_summary
