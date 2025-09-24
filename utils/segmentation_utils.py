import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def perform_segmentation(df, num_clusters=4):
    """
    Segments the dataframe using KMeans clustering.
    Automatically one-hot encodes non-numeric columns.
    """
    # One-hot encode all categorical columns
    processed_df = pd.get_dummies(df)

    if processed_df.empty:
        raise ValueError("No usable data found for clustering after encoding.")

    # Fit KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(processed_df)

    # Add cluster labels back to original dataframe
    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    return clustered_df, labels

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

def plot_segmentation(clustered_df, labels):
    """
    Plots customer clusters in 2D using first two numeric features.

    Parameters:
        clustered_df (DataFrame): The clustered customer data.
        labels (array-like): Cluster labels for each point.

    Returns:
        matplotlib.figure.Figure: The segmentation plot.
    """
    # Select the first two numeric columns for plotting
    numeric_cols = clustered_df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        raise ValueError("Need at least two numeric columns to plot segmentation.")

    x_col, y_col = numeric_cols[:2]
    clustered_df["Cluster"] = labels

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=clustered_df,
        x=x_col,
        y=y_col,
        hue="Cluster",
        palette="tab10",
        s=60,
        edgecolor="black"
    )
    plt.title("Customer Segmentation Clusters")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="Cluster")
    plt.tight_layout()

    return plt.gcf()
