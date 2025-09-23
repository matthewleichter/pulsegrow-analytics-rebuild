#dependence is scripts/causal_inference.py

# utils/causal_inference_utils.py

import pandas as pd
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def run_causal_model(df):
    # Expect 'treatment', 'outcome', and features in df
    treatment = df["treatment"].values
    outcome = df["outcome"].values
    features = df.drop(columns=["treatment", "outcome"])

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        features, treatment, outcome, test_size=0.2, random_state=42
    )

    model_y = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    model_t = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)

    est = DML(model_y=model_y, model_t=model_t, random_state=0)
    est.fit(Y_train, T_train, X=X_train)

    treatment_effects = est.effect(X_test)

    results_df = X_test.copy()
    results_df["estimated_effect"] = treatment_effects

    return results_df.head(10)
