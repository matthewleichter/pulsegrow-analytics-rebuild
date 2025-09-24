import streamlit as st
from scripts import (
    ab_test_analysis, anomaly_detection, causal_inference, churn_analysis,
    funnel_analysis, llm_interpreter, marketing_mix_model, revenue_forecast,
    segmentation_model, kaplan_meier_model, timeseries_forecast, usage_forecasting
)
from scripts import prepare_visuals 
import os

st.set_page_config(page_title="PulseGrow Analytics", layout="wide")

# ✅ Sidebar control to optionally refresh visualizations
st.sidebar.title("🔄 Data Preprocessing")
refresh_charts = st.sidebar.checkbox("Regenerate all charts", value=False)

# ✅ Run prepare_charts() on first launch or if user asks
CHART_FLAG_FILE = "assets/visuals/.charts_ready"

if refresh_charts or not os.path.exists(CHART_FLAG_FILE):
    with st.spinner("Generating visualizations..."):
        prepare_visuals.prepare_charts()
        with open(CHART_FLAG_FILE, "w") as f:
            f.write("ready")
        st.sidebar.success("✅ Charts generated.")
else:
    st.sidebar.info("✅ Cached charts ready.")
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
    kaplan_meier_model.run_kaplan_meier()

elif tab == "Segmentation Model":
    segmentation_model.run_segmentation_model()

elif tab == "Causal Inference":
    causal_inference.run_causal_inference()

elif tab == "A/B Test Analysis":
    ab_test_analysis.run_ab_test_analysis()

elif tab == "Anomaly Detection":
    anomaly_detection.run_anomaly_detection()

elif tab == "Revenue Forecast":
    revenue_forecast.run_revenue_forecast()

elif tab == "Marketing Mix Modeling":
    marketing_mix_model.run_marketing_mix_model()

elif tab == "LLM Interpreter":
    llm_interpreter.run_llm_interpreter()

elif tab == "Funnel Analysis":
    funnel_analysis.run_funnel_analysis()
