import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_churn_model(df, target_column='churn'):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Optional: encode categorical columns if present
    X = pd.get_dummies(X)

    # Align columns in case dummy columns mismatch during predict time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report

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

import matplotlib.pyplot as plt

def plot_churn_distribution(df):
    if 'churn' not in df.columns:
        raise ValueError("DataFrame must contain a 'churn' column.")

    fig, ax = plt.subplots()
    df['churn'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title('Churn Distribution')
    ax.set_xlabel('Churn (0 = No, 1 = Yes)')
    ax.set_ylabel('Count')
    return fig


