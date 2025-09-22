
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.churn_utils import *
from utils.usage_utils import *
from utils.forecast_utils import *
from utils.survival_utils import *
from utils.segmentation_utils import *
from utils.funnel_utils import *
from utils.abtest_utils import *
from utils.causal_utils import *
from utils.anomaly_utils import *
from utils.revenue_utils import *
from utils.insight_utils import *

st.set_page_config(page_title="PulseGrow Analytics", layout="wide")

st.title("📊 PulseGrow Analytics Platform")

# Load data
@st.cache_data
def load_data():
    events = pd.read_csv("data/events.csv")
    users = pd.read_csv("data/users.csv")
    transactions = pd.read_csv("data/transactions.csv")
    churn_labels = pd.read_csv("data/churn_labels.csv")
    usage_logs = pd.read_csv("data/usage_logs.csv")
    retention_data = pd.read_csv("data/retention_data.csv")
    segmentation_labels = pd.read_csv("data/segmentation_labels.csv")
    funnel_steps = pd.read_csv("data/funnel_steps.csv")
    ab_test_results = pd.read_csv("data/ab_test_results.csv")
    causal_treatments = pd.read_csv("data/causal_treatments.csv")
    anomaly_logs = pd.read_csv("data/anomaly_logs.csv")
    marketing_spend = pd.read_csv("data/marketing_spend.csv")
    product_features = pd.read_csv("data/product_features.csv")
    return {
        "events": events,
        "users": users,
        "transactions": transactions,
        "churn_labels": churn_labels,
        "usage_logs": usage_logs,
        "retention_data": retention_data,
        "segmentation_labels": segmentation_labels,
        "funnel_steps": funnel_steps,
        "ab_test_results": ab_test_results,
        "causal_treatments": causal_treatments,
        "anomaly_logs": anomaly_logs,
        "marketing_spend": marketing_spend,
        "product_features": product_features
    }

data = load_data()

tabs = st.tabs([
    "Churn Prediction",
    "Usage Forecast",
    "TimesNet Forecast",
    "Retention (Kaplan-Meier)",
    "Segmentation",
    "Funnel Analysis",
    "A/B Testing",
    "Causal Inference",
    "Anomaly Detection",
    "Revenue Forecast",
    "Insight Generator"
])

with tabs[0]:
    st.subheader("📉 Churn Prediction Model")
    churn_model(data["users"], data["churn_labels"])

with tabs[1]:
    st.subheader("📈 Usage Forecasting (Smoothed + CI)")
    usage_forecast(data["usage_logs"])

with tabs[2]:
    st.subheader("🧠 TimesNet Usage Forecast (Next 5 Points)")
    timesnet_forecast(data["usage_logs"])

with tabs[3]:
    st.subheader("📊 Retention Analysis (Kaplan-Meier)")
    retention_km(data["retention_data"])

with tabs[4]:
    st.subheader("👥 Behavioral Segmentation")
    segmentation_model(data["usage_logs"], data["segmentation_labels"])

with tabs[5]:
    st.subheader("🔍 Funnel Conversion Analysis")
    funnel_analysis(data["funnel_steps"])

with tabs[6]:
    st.subheader("🧪 A/B Test Simulation and Results")
    ab_test_module(data["ab_test_results"])

with tabs[7]:
    st.subheader("⚖️ Causal Inference (DoWhy / EconML)")
    causal_inference_module(data["causal_treatments"])

with tabs[8]:
    st.subheader("🚨 Anomaly Detection (Isolation Forest)")
    anomaly_detection(data["anomaly_logs"])

with tabs[9]:
    st.subheader("💰 Revenue Forecasting (Prophet/ARIMA)")
    revenue_forecast(data["transactions"], data["marketing_spend"])

with tabs[10]:
    st.subheader("🧠 AI Insight Generator (LLM-Augmented)")
    insight_generator(data["events"], data["product_features"])
