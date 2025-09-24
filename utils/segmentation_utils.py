import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def perform_segmentation(df, num_clusters=4):
    """
    Perform KMeans clustering on the numeric columns of the dataframe.

    Parameters:
        df (pd.DataFrame): The input data.
        num_clusters (int): The number of clusters to use.

    Returns:
        clustered_df (pd.DataFrame): The original DataFrame with an added 'cluster' column.
        labels (np.ndarray): The array of cluster labels.
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 1:
        raise ValueError("No numeric columns found for clustering.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    return clustered_df, labels

def plot_segmentation(df, labels):
    """
    Plot the clusters using the first two numeric columns.

    Parameters:
        df (pd.DataFrame): The clustered data.
        labels (np.ndarray): The cluster labels.

    Returns:
        matplotlib.figure.Figure: The figure to render with Streamlit.
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns to plot clusters.")

    numeric_df["cluster"] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=numeric_df.columns[0],
        y=numeric_df.columns[1],
        hue="cluster",
        palette="tab10",
        data=numeric_df,
        s=80,
        alpha=0.8
    )
    plt.title("Customer Segmentation Clusters")
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.legend(title="Cluster")
    plt.tight_layout()

    return plt.gcf()
