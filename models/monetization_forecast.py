import pandas as pd
from prophet import Prophet

def forecast_revenue(df):
    df = df.rename(columns={"date": "ds", "revenue": "y"})
    model = Prophet()
    model.fit(df)
    forecast = model.predict(df)
    return forecast