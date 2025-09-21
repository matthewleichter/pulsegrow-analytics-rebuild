import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def load_usage_data():
    return pd.read_csv('data/usage_logs.csv')

def plot_smoothed_usage(df):
    df = df.sort_values('timestamp')
    usage = df['usage_count'].values
    smoothed = gaussian_filter1d(usage, sigma=2)
    ci = 1.96 * np.std(usage)/np.sqrt(len(usage))
    plt.plot(smoothed, label='Smoothed')
    plt.fill_between(range(len(smoothed)), smoothed - ci, smoothed + ci, alpha=0.3)
    plt.title("Smoothed Usage with Confidence Interval")
    return plt.gcf()