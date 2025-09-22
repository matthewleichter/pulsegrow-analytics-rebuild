
import matplotlib.pyplot as plt

def plot_distribution(data, column, title="Distribution"):
    plt.figure(figsize=(8, 4))
    data[column].hist(bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
