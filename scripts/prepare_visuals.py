import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import (
    plot_usage_forecast,
    plot_churn_results,
    plot_segmentation_clusters
)
from utils.revenue_utils import generate_revenue_forecast_plot
from utils.segmentation_utils import load_segmentation_results
from utils.usage_utils import load_usage_forecast_data
from utils.churn_utils import load_churn_predictions

OUTPUT_DIR = "assets/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def maybe_save_plot(fig, path):
    """Only saves plot if it doesn't already exist"""
    if not os.path.exists(path):
        fig.savefig(path, bbox_inches="tight")
        print(f"✅ Generated: {path}")
    else:
        print(f"⏩ Skipped (already exists): {path}")
    plt.close(fig)

def prepare_charts():
    print("🎨 Preparing visualizations...")

    try:
        usage_data = load_usage_forecast_data()
        usage_fig = plot_usage_forecast(usage_data)
        maybe_save_plot(usage_fig, os.path.join(OUTPUT_DIR, "usage_forecast.png"))
    except Exception as e:
        print(f"❌ Error preparing usage forecast plot: {e}")

    try:
        churn_data = load_churn_predictions()
        churn_fig = plot_churn_results(churn_data)
        maybe_save_plot(churn_fig, os.path.join(OUTPUT_DIR, "churn_prediction.png"))
    except Exception as e:
        print(f"❌ Error preparing churn prediction plot: {e}")

    try:
        revenue_fig = generate_revenue_forecast_plot()
        maybe_save_plot(revenue_fig, os.path.join(OUTPUT_DIR, "revenue_forecast.png"))
    except Exception as e:
        print(f"❌ Error preparing revenue forecast plot: {e}")

    try:
        segments = load_segmentation_results()
        seg_fig = plot_segmentation_clusters(segments)
        maybe_save_plot(seg_fig, os.path.join(OUTPUT_DIR, "user_segmentation.png"))
    except Exception as e:
        print(f"❌ Error preparing segmentation plot: {e}")

    print("✅ Done preparing all visualizations.")

if __name__ == "__main__":
    prepare_charts()

    print("✅ All visualizations saved.")

if __name__ == "__main__":
    prepare_charts()
