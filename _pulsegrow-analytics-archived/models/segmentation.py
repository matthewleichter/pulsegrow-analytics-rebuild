from sklearn.cluster import KMeans

def segment_users(df):
    kmeans = KMeans(n_clusters=3)
    df["segment"] = kmeans.fit_predict(df[["age", "income", "activity"]])
    return df