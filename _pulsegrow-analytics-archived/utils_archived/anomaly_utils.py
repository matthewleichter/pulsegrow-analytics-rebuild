from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    clf = IsolationForest(contamination=0.01)
    preds = clf.fit_predict(data)
    data['anomaly'] = preds
    return data