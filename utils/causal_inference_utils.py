#dependence is scripts/causal_inference.py

# utils/causal_inference_utils.py

import pandas as pd
from causalml.inference.meta import BaseXClassifier
from sklearn.model_selection import train_test_split

def run_causal_model(df):
    # Expect 'treatment', 'outcome', and features
    treatment = df["treatment"].values
    outcome = df["outcome"].values
    features = df.drop(columns=["treatment", "outcome"])

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        features, treatment, outcome, test_size=0.2, random_state=42
    )

    learner = BaseXClassifier()
    learner.fit(X=X_train.values, treatment=T_train, y=Y_train)
    uplift = learner.predict(X_test.values)

    results_df = X_test.copy()
    results_df["uplift_score"] = uplift

    return results_df.head(10)
