
import streamlit as st
import pandas as pd
from utils.data_loader import load_all_data
from utils.plot_helpers import (
    plot_churn_analysis,
    plot_usage_trends,
    plot_kaplan_meier,
    plot_timesnet_forecast,
    plot_ab_test,
    plot_anomalies,
    plot_causal_inference,
    plot_funnel_analysis,
    plot_revenue_forecast,
    plot_segmentation,
    plot_llm_interpreter
)

st.set_page_config(page_title="PulseGrow Analytics", layout="wide")

st.title("📊 PulseGrow Analytics Dashboard")
st.markdown("Explore all 11 analytics modules below:")

# Load all data
data = load_all_data()

# Define tabs for each module
tabs = st.tabs([
    "Churn Prediction", "Usage Forecast", "TimesNet Forecast", "Retention (Survival)",
    "A/B Testing", "Anomaly Detection", "Causal Inference",
    "Funnel Analysis", "Revenue Forecast", "Segmentation", "LLM Interpreter"
])

# Churn Prediction
with tabs[0]:
    st.subheader("Churn Prediction")
    plot_churn_analysis(data)

# Usage Forecast
with tabs[1]:
    st.subheader("Usage Forecast")
    plot_usage_trends(data)

# TimesNet Forecast
with tabs[2]:
    st.subheader("TimesNet - Next 5 Points Forecast")
    plot_timesnet_forecast(data)

# Retention (Kaplan-Meier)
with tabs[3]:
    st.subheader("User Retention - Kaplan-Meier Survival Curve")
    plot_kaplan_meier(data)

# A/B Testing
with tabs[4]:
    st.subheader("A/B Test Results")
    plot_ab_test(data)

# Anomaly Detection
with tabs[5]:
    st.subheader("Anomaly Detection")
    plot_anomalies(data)

# Causal Inference
with tabs[6]:
    st.subheader("Causal Inference Analysis")
    plot_causal_inference(data)

# Funnel Analysis
with tabs[7]:
    st.subheader("Funnel Drop-Off Analysis")
    plot_funnel_analysis(data)

# Revenue Forecast
with tabs[8]:
    st.subheader("Revenue & Monetization Forecast")
    plot_revenue_forecast(data)

# Segmentation
with tabs[9]:
    st.subheader("Behavioral Segmentation")
    plot_segmentation(data)

# LLM Interpreter
with tabs[10]:
    st.subheader("LLM Model Insights")
    plot_llm_interpreter(data)
