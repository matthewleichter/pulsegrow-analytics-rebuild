#dependence is scripts/causal_inference.py

import pandas as pd
import numpy as np
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "data/causal_treatments.csv"
OUTPUT_PATH = "outputs/econml_causal_effects.csv"

def run_causal_inference():
    print("🔍 Running causal inference using EconML...")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"📥 Loaded {len(df)} rows from {DATA_PATH}")

    # Define treatment, outcome, and controls
    treatment = df["treatment"].values
    outcome = df["outcome"].values
    features = df.drop(columns=["treatment", "outcome"])

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        features, treatment, outcome, test_size=0.2, random_state=42
    )

    # Initialize EconML DML Estimator
    print("🧠 Training Double ML estimator...")
    model_y = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    model_t = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

    est = DML(model_y=model_y, model_t=model_t, random_state=0)
    est.fit(Y_train, T_train, X=X_train)

    # Estimate treatment effect
    print("📈 Estimating treatment effects...")
    treatment_effects = est.effect(X_test)

    # Save results
    results_df = X_test.copy()
    results_df["treatment_effect"] = treatment_effects

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Causal inference completed. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_causal_inference()
