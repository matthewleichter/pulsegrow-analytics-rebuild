import pandas as pd
import streamlit as st
from utils.churn_utils import analyze_churn

def run_churn_analysis():
    st.title("Churn Analysis")
    uploaded_file = st.file_uploader("Upload churn data CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        churn_metrics = analyze_churn(data)
        st.write("Churn Metrics:", churn_metrics)
