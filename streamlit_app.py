import streamlit as st
from scripts import (
    ab_test_analysis, anomaly_detection, causal_inference, churn_analysis,
    funnel_analysis, llm_interpreter, marketing_mix_model, revenue_forecast,
    segmentation_model, survival_analysis, timeseries_forecast, usage_forecasting
)

st.set_page_config(page_title="PulseGrow Analytics", layout="wide")
st.title("📊 PulseGrow Analytics Dashboard")
st.markdown("""
Welcome to **PulseGrow**, your unified analytics hub for behavioral forecasting, segmentation, causal inference, A/B testing, and more. 
Select a model tab below to begin your analysis.
""")

# Sidebar Styling and Info
with st.sidebar:
    st.header("🔧 Select Analysis Module")
    st.markdown("---")
    tab = st.radio(
        label="Choose a model",
        options=[
            "Churn Analysis", "Usage Forecasting", "Timeseries Forecast",
            "Survival Analysis", "Segmentation Model", "Causal Inference",
            "A/B Test Analysis", "Anomaly Detection", "Revenue Forecast",
            "Marketing Mix Modeling", "LLM Interpreter", "Funnel Analysis"
        ]
    )
    st.markdown("---")
    st.caption("ℹ️ Built with ❤️ by Matthew Leichter on Leprechaun OS, the only existing LLM Based Operating System")
    st.caption("Matthew Leichter | matthew.leichter@gmail.com | (323) 303-8062 | https://matthewleichter.github.io")

# Tab Logic
if selected == "Churn Analysis":
    churn_analysis.run_churn_analysis()

elif tab == "Usage Forecasting":
    usage_forecasting.run_usage_forecasting()

elif tab == "Timeseries Forecast":
    timeseries_forecast.run_timeseries_forecast()

elif tab == "Survival Analysis":
    survival_analysis.run_survival_analysis()

elif tab == "Segmentation Model":
    segmentation_model.run_segmentation_model()

elif tab == "Causal Inference":
    causal_inference.run_causal_inference()

elif tab == "A/B Test Analysis":
    ab_test_analysis.run_ab_test_analysis()

elif tab == "Anomaly Detection":
    anomaly_detection.run_anomaly_detection()

elif tab == "Revenue Forecast":
    revenue_forecast.run_revenue_forecasting()

elif tab == "Marketing Mix Modeling":
    marketing_mix_model.run_marketing_mix_model()

elif tab == "LLM Interpreter":
    llm_interpreter.run_llm_interpreter()

elif tab == "Funnel Analysis":
    funnel_analysis.run_funnel_analysis()
