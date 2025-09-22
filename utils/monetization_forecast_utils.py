from prophet import Prophet

def forecast_revenue(data):
    df = data.rename(columns={'date': 'ds', 'revenue': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast