import xgboost as xgb
import pandas as pd
import numpy as np

def TimesNetPredictor(df, target_col='usage', forecast_horizon=5):
    """
    Train an XGBoost regressor on past usage and predict the next `forecast_horizon` values.
    Assumes the input df has a time-series index and a column named 'usage'.
    """
    df = df[[target_col]].dropna().copy()
    df['lag1'] = df[target_col].shift(1)
    df['lag2'] = df[target_col].shift(2)
    df['lag3'] = df[target_col].shift(3)

    df.dropna(inplace=True)

    X = df[['lag1', 'lag2', 'lag3']]
    y = df[target_col]

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    # Generate forecasts iteratively
    last_values = df[target_col].iloc[-3:].tolist()
    preds = []

    for _ in range(forecast_horizon):
        input_vec = np.array(last_values[-3:]).reshape(1, -1)
        pred = model.predict(input_vec)[0]
        preds.append(pred)
        last_values.append(pred)

    return preds
