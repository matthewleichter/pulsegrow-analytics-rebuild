import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_causal_data():
    return pd.read_csv('data/causal_treatments.csv')

def estimate_treatment_effect(df):
    model = LinearRegression()
    model.fit(df[['treatment']], df['outcome'])
    effect = model.coef_[0]
    plt.bar(['Treatment Effect'], [effect])
    plt.title("Estimated Treatment Effect")
    return plt.gcf()