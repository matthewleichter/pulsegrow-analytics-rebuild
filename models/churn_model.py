import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_churn_model(df):
    X = df[["usage", "age", "tenure"]]
    y = df["churn"]
    model = LogisticRegression()
    model.fit(X, y)
    return model