
import pandas as pd
import numpy as np

def compute_confidence_interval(df, ci=0.95):
    usage = df['usage']
    mean = usage.mean()
    std = usage.std()
    n = len(usage)
    margin = 1.96 * (std / np.sqrt(n))  # 95% CI with z-score
    return mean - margin, mean + margin

def forecast_basic(df, steps=7):
    last_value = df['usage'].iloc[-1]
    return pd.DataFrame({'day': range(len(df), len(df)+steps), 'forecast': [last_value]*steps})
