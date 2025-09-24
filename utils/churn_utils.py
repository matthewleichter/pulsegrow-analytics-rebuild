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

def analyze_churn(df):
    """
    Performs basic churn analysis and returns churn rate and summary statistics.
    """
    if "churn" not in df.columns:
        raise ValueError("DataFrame must contain a 'churn' column.")

    churn_rate = df["churn"].mean()
    total = len(df)
    churned = df["churn"].sum()
    retained = total - churned

    summary = {
        "Total Users": total,
        "Churned Users": int(churned),
        "Retained Users": int(retained),
        "Churn Rate": round(churn_rate * 100, 2)
    }

    return summary

def load_churn_predictions(path="data/churn_predictions.csv"):
    """
    Loads predicted churn probabilities from CSV.
    """
    return pd.read_csv(path)
