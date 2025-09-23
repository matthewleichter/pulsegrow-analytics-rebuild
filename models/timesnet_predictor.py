# models/timesnet_predictor.py

import xgboost as xgb
import numpy as np
import pandas as pd

class XGBoostPredictor:
    def __init__(self, input_window=30, forecast_horizon=5):
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

    def _create_lag_features(self, series):
        df = pd.DataFrame(series, columns=["usage"])
        for i in range(1, 4):
            df[f"lag{i}"] = df["usage"].shift(i)
        return df.dropna()

    def fit(self, series):
        df = self._create_lag_features(series)
        X = df[["lag1", "lag2", "lag3"]]
        y = df["usage"]
        self.model.fit(X, y)

    def predict(self, series):
        self.fit(series)

        last_values = list(series[-3:])
        preds = []

        for _ in range(self.forecast_horizon):
            input_array = np.array(last_values[-3:]).reshape(1, -1)
            next_val = self.model.predict(input_array)[0]
            preds.append(next_val)
            last_values.append(next_val)

        return preds
