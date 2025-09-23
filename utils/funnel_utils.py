import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_funnel_report(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with 'stage' and 'user_id' and returns conversion rates.
    Assumes each user_id appears once per stage.
    """
    funnel_counts = data.groupby('stage')['user_id'].nunique().reset_index()
    funnel_counts = funnel_counts.sort_values('stage', ascending=True).reset_index(drop=True)

    funnel_counts['conversion_rate'] = funnel_counts['user_id'] / funnel_counts['user_id'].iloc[0]
    funnel_counts.columns = ['Stage', 'Users', 'Conversion Rate']
    return funnel_counts

def plot_funnel_chart(funnel_df: pd.DataFrame):
    """
    Plot a funnel chart showing drop-off at each stage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=funnel_df, x='Stage', y='Users', ax=ax, palette='Blues_d')
    ax.set_title("Funnel Drop-off by Stage")
    ax.set_ylabel("Number of Users")
    ax.set_xlabel("Funnel Stage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_conversion_rates(funnel_df: pd.DataFrame):
    """
    Plot conversion rate at each stage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=funnel_df, x='Stage', y='Conversion Rate', marker='o', ax=ax, color='green')
    ax.set_title("Conversion Rate by Stage")
    ax.set_ylabel("Conversion Rate")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Funnel Stage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
