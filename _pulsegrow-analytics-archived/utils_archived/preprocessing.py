import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_numerical(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')