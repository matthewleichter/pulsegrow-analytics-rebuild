# prepare_visuals.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import plot_usage_forecast, plot_churn_results, plot_segmentation_clusters
from utils.revenue_utils import generate_revenue_forecast_plot
from utils.segmentation_utils import load_segmentation_results
from utils.usage_utils import load_usage_forecast_data
from utils.churn_utils import load_churn_predictions

OUTPUT_DIR = "assets/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_charts():
    print("🎨 Preparing visualizations...")

    # 1. Usage Forecast Visualization
    usage_data = load_usage_forecast_data()
    usage_fig = plot_usage_forecast(usage_data)
    usage_path = os.path.join(OUTPUT_DIR, "usage_forecast.png")
    usage_fig.savefig(usage_path)
    print(f"📊 Saved usage forecast plot to {usage_path}")

    # 2. Churn Prediction Visualization
    churn_data = load_churn_predictions()
    churn_fig = plot_churn_results(churn_data)
    churn_path = os.path.join(OUTPUT_DIR, "churn_prediction.png")
    churn_fig.savefig(churn_path)
    print(f"📉 Saved churn prediction plot to {churn_path}")

    # 3. Revenue Forecast Visualization
    revenue_fig = generate_revenue_forecast_plot()
    revenue_path = os.path.join(OUTPUT_DIR, "revenue_forecast.png")
    revenue_fig.savefig(revenue_path)
    print(f"💰 Saved revenue forecast plot to {revenue_path}")

    # 4. Segmentation Visualization
    segments = load_segmentation_results()
    seg_fig = plot_segmentation_clusters(segments)
    seg_path = os.path.join(OUTPUT_DIR, "user_segmentation.png")
    seg_fig.savefig(seg_path)
    print(f"🧩 Saved segmentation plot to {seg_path}")

    print("✅ All visualizations saved.")

if __name__ == "__main__":
    prepare_charts()
