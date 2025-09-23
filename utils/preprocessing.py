import pandas as pd
import numpy as np


def preprocess_timeseries(
    df: pd.DataFrame,
    usage_col: str = "usage",
    timestamp_col: str = "timestamp",
    resample_freq: str = None,
    rolling_window: int = 3,
    create_lags: int = 3,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for time series forecasting.

    Args:
        df (pd.DataFrame): Input dataframe with usage and timestamp.
        usage_col (str): Name of the usage column.
        timestamp_col (str): Name of the timestamp column.
        resample_freq (str): Optional. e.g., 'D' for daily, 'H' for hourly.
        rolling_window (int): Window size for optional smoothing.
        create_lags (int): Number of lag features to create.
        normalize (bool): Whether to normalize the usage column.

    Returns:
        pd.DataFrame: Processed dataframe with lag features and smoothed values.
    """
    df = df.copy()

    # Convert timestamp to datetime if necessary
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.set_index(timestamp_col, inplace=True)

    # Sort index to ensure time ordering
    df.sort_index(inplace=True)

    # Fill missing usage values
    df[usage_col] = df[usage_col].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    # Resample if specified
    if resample_freq:
        df = df.resample(resample_freq).mean()

    # Apply smoothing using rolling window
    if rolling_window > 1:
        df[f"{usage_col}_smoothed"] = df[usage_col].rolling(window=rolling_window, min_periods=1).mean()
    else:
        df[f"{usage_col}_smoothed"] = df[usage_col]

    # Normalize
    if normalize:
        mean = df[f"{usage_col}_smoothed"].mean()
        std = df[f"{usage_col}_smoothed"].std()
        df[f"{usage_col}_normalized"] = (df[f"{usage_col}_smoothed"] - mean) / std
    else:
        df[f"{usage_col}_normalized"] = df[f"{usage_col}_smoothed"]

    # Create lag features
    for i in range(1, create_lags + 1):
        df[f"lag_{i}"] = df[f"{usage_col}_normalized"].shift(i)

    df.dropna(inplace=True)  # Remove rows with incomplete lag features

    return df


def extract_usage_series(df: pd.DataFrame, usage_col: str = "usage", input_window: int = 30) -> np.ndarray:
    """
    Extracts the most recent usage values for forecasting.

    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        usage_col (str): The normalized usage column to extract.
        input_window (int): Number of recent values to extract.

    Returns:
        np.ndarray: Numpy array of the most recent usage values.
    """
    col = f"{usage_col}_normalized" if f"{usage_col}_normalized" in df.columns else usage_col
    return df[col].values[-input_window:]
