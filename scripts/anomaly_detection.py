import pandas as pd
import streamlit as st
from utils.anomaly_utils import detect_anomalies, plot_anomalies

def run_anomaly_detection():
    st.title("Anomaly Detection Module")

    uploaded_file = st.file_uploader("Upload CSV with 'usage' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'usage' in df.columns:
            st.subheader("📊 Raw Data")
            st.dataframe(df.head())

            df, model = detect_anomalies(df)
            st.subheader("🚨 Anomaly Detection Results")
            st.dataframe(df.head())

            st.subheader("📉 Anomaly Visualization")
            st.pyplot(plot_anomalies(df))

        else:
            st.error("❌ The uploaded CSV must contain a 'usage' column.")
