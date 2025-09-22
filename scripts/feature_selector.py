
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, top_n=10):
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]
    return X.columns[indices].tolist()
