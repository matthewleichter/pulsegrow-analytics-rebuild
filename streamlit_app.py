import streamlit as st
from scripts import (
    ab_test_analysis, anomaly_detection, causal_inference, churn_prediction,
    funnel_analysis, llm_interpreter, marketing_mix_model, revenue_forecast,
    segmentation_model, survival_analysis, timeseries_forecast, usage_forecast
)

st.set_page_config(page_title="PulseGrow Analytics", layout="wide")
st.sidebar.title("📊 PulseGrow Analytics")
app_mode = st.sidebar.selectbox("Choose a Module", [
    "User Churn Prediction", "Usage Forecast", "TimesNet Forecast",
    "Retention (Kaplan-Meier)", "Causal Inference", "A/B Test Analysis",
    "Funnel Analysis", "Revenue Forecast", "Behavioral Segmentation",
    "Anomaly Detection", "LLM Insight Generator", "Marketing Mix Model"
])

if app_mode == "User Churn Prediction":
    churn_prediction.app()
elif app_mode == "Usage Forecast":
    usage_forecast.app()
elif app_mode == "TimesNet Forecast":
    timeseries_forecast.app()
elif app_mode == "Retention (Kaplan-Meier)":
    survival_analysis.app()
elif app_mode == "Causal Inference":
    causal_inference.app()
elif app_mode == "A/B Test Analysis":
    ab_test_analysis.app()
elif app_mode == "Funnel Analysis":
    funnel_analysis.app()
elif app_mode == "Revenue Forecast":
    revenue_forecast.app()
elif app_mode == "Behavioral Segmentation":
    segmentation_model.app()
elif app_mode == "Anomaly Detection":
    anomaly_detection.app()
elif app_mode == "LLM Insight Generator":
    llm_interpreter.app()
elif app_mode == "Marketing Mix Model":
    marketing_mix_model.app()