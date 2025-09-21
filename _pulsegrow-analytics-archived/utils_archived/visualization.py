import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df, column, title='Distribution'):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()

def plot_correlation_matrix(df, title='Correlation Matrix'):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.title(title)
    plt.show()