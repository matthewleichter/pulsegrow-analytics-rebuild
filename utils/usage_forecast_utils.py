import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth_usage(data, frac=0.1):
    smoothed = lowess(data['usage'], data['date'], frac=frac, return_sorted=False)
    data['smoothed'] = smoothed
    return data