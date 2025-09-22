import pandas as pd

def create_time_features(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df['hour'] = df[time_column].dt.hour
    df['day'] = df[time_column].dt.day
    df['weekday'] = df[time_column].dt.weekday
    return df