import pandas as pd
import plotly.express as px

def load_funnel_data():
    return pd.read_csv('data/funnel_steps.csv')

def plot_funnel(df):
    steps = df['step'].value_counts().sort_index()
    fig = px.funnel_area(names=steps.index, values=steps.values)
    return fig