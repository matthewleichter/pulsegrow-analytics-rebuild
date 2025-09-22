import pandas as pd
import matplotlib.pyplot as plt

def load_ab_test_data():
    return pd.read_csv('data/ab_test_results.csv')

def plot_ab_test(df):
    control = df[df['group'] == 'control']['conversion']
    treatment = df[df['group'] == 'treatment']['conversion']
    plt.hist([control, treatment], label=['Control', 'Treatment'], bins=20)
    plt.legend()
    plt.title("A/B Test Conversion Rates")
    return plt.gcf()