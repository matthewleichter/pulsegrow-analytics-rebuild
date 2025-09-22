# run_all_models.py

from churn_model import run_churn_model
from usage_forecast import run_usage_forecast
from timesnet_predictor import run_timesnet_forecast
from retention_km import run_kaplan_meier_model
from revenue_forecast import run_revenue_forecast
from causal_inference import run_causal_model
from segmentation_model import run_segmentation_model
from funnel_analysis import run_funnel_analysis
from llm_interpretation import run_llm_interpreter
from anomaly_detection import run_anomaly_detection

def run_models():
    print("🔁 Running all model pipelines...\n")

    print("📌 Running Churn Model...")
    run_churn_model()

    print("📌 Running Usage Forecast Model...")
    run_usage_forecast()

    print("📌 Running TimesNet Forecast Model...")
    run_timesnet_forecast()

    print("📌 Running Retention Model (Kaplan-Meier)...")
    run_kaplan_meier_model()

    print("📌 Running Revenue Forecast...")
    run_revenue_forecast()

    print("📌 Running Causal Inference Model...")
    run_causal_model()

    print("📌 Running Segmentation Model...")
    run_segmentation_model()

    print("📌 Running Funnel Analysis...")
    run_funnel_analysis()

    print("📌 Running LLM Interpreter...")
    run_llm_interpreter()

    print("📌 Running Anomaly Detection...")
    run_anomaly_detection()

    print("\n✅ All models executed successfully.")

if __name__ == "__main__":
    run_models()
