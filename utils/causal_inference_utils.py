#dependence is scripts/causal_inference.py

# utils/causal_inference_utils.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def run_causal_model(df):
    if "treatment" not in df.columns or "outcome" not in df.columns:
        return "❌ Dataset must include 'treatment' and 'outcome' columns."

    features = df.drop(columns=["treatment", "outcome"])
    treatment = df["treatment"]
    outcome = df["outcome"]

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        features, treatment, outcome, test_size=0.2, random_state=42
    )

    # Model the outcome given features and treatment as a feature
    X_train_full = pd.concat([X_train, T_train], axis=1)
    X_test_full = pd.concat([X_test, T_test], axis=1)

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_full, Y_train)
    preds = model.predict(X_test_full)

    report = classification_report(Y_test, preds, output_dict=True)
    return pd.DataFrame(report).transpose().round(2)
