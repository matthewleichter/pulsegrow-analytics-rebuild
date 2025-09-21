import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_churn_model(data):
    features = data.drop(columns=['churn_label'])
    labels = data['churn_label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model

def predict_churn(model, new_data):
    return model.predict_proba(new_data)[:, 1]