import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def load_retention_data():
    return pd.read_csv('data/retention_data.csv')

def plot_kaplan_meier(df):
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], event_observed=df["churned"])
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    return plt.gcf()