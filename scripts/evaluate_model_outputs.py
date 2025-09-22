# evaluate_model_outputs.py

import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

OUTPUT_DIR = "outputs"
EVALUATION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "evaluation_summary.json")

def evaluate_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def evaluate_regression(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }

def evaluate_outputs():
    evaluations = {}

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(OUTPUT_DIR, filename)
            df = pd.read_csv(filepath)

            model_name = filename.replace(".csv", "")
            if "label" in df.columns and "prediction" in df.columns:
                y_true = df["label"]
                y_pred = df["prediction"]

                if df["label"].nunique() <= 10 and df["label"].dtype in ['int64', 'int32']:
                    evaluations[model_name] = evaluate_classification(y_true, y_pred)
                else:
                    evaluations[model_name] = evaluate_regression(y_true, y_pred)

    # Save to JSON
    with open(EVALUATION_RESULTS_FILE, "w") as f:
        json.dump(evaluations, f, indent=4)

    print(f"✅ Model evaluations saved to {EVALUATION_RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_outputs()
