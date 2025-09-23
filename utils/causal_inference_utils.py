#dependence is scripts/causal_inference.py

# utils/causal_inference_utils.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_causal_model(df):
    # Split data into treatment, control, and outcome
    treatment = df["treatment"]
    outcome = df["outcome"]
    features = df.drop(columns=["treatment", "outcome"])

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        features, treatment, outcome, test_size=0.2, random_state=42
    )

    # Train separate models for treated and untreated
    model_treated = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    model_control = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

    # Fit on treated and control groups separately
    model_treated.fit(X_train[T_train == 1], Y_train[T_train == 1])
    model_control.fit(X_train[T_train == 0], Y_train[T_train == 0])

    # Estimate treatment effect: difference in predicted outcomes
    mu1 = model_treated.predict(X_test)
    mu0 = model_control.predict(X_test)
    treatment_effect = mu1 - mu0

    # Return results
    results_df = X_test.copy()
    results_df["estimated_effect"] = treatment_effect
    return results_df.head(10)
