import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

def load_churn_data():
    return pd.read_csv('data/churn_labels.csv')

def train_churn_model(X, y):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    return model

def plot_churn_predictions(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    plt.hist(preds, bins=50, alpha=0.7)
    plt.title("Churn Prediction Probability")
    return plt.gcf()