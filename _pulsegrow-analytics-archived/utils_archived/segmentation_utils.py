from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_kmeans_segmentation(data, n_clusters=4):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)
    return reduced, labels