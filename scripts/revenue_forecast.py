# revenue_forecast.py

import pandas as pd
import matplotlib.pyplot as plt
from utils.revenue_utils import preprocess_revenue_data, forecast_revenue
from utils.visualization import plot_forecast_with_confidence

def run_revenue_forecast():
    # Load data
    transactions = pd.read_csv("data/transactions.csv")
    marketing = pd.read_csv("data/marketing_spend.csv")

    # Preprocess and merge datasets
    revenue_df = preprocess_revenue_data(transactions, marketing)

    # Forecast revenue (e.g., using Prophet, TimesNet, ARIMA)
    forecast_df = forecast_revenue(revenue_df)

    # Plot results
    fig = plot_forecast_with_confidence(forecast_df)
    fig.suptitle("Revenue Forecast vs Actual")

    # Show or save
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_revenue_forecast()
