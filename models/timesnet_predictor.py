from utils.timesnet_utils import TimesNetPredictor
from utils.preprocessing import preprocess_timeseries

def predict_next_5_timesnet(df):
    """
    Predict the next 5 usage values using the TimesNet model.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing a 'usage' column with timestamp index.

    Returns:
        np.ndarray: Array of 5 predicted usage values.
    """
    print("🔮 Running TimesNet forecast for next 5 usage points...")

    # Preprocess the time series
    ts_data = preprocess_timeseries(df)

    # Instantiate the model
    model = TimesNetPredictor(input_window=30, forecast_horizon=5)

    # Fit model on full data
    model.fit(ts_data)

    # Predict the next 5 points
    forecast = model.predict(ts_data)

    return forecast
