import pandas as pd
import streamlit as st
from utils.anomaly_utils import detect_anomalies

def run_anomaly_detection():
    st.title("Anomaly Detection")
    uploaded_file = st.file_uploader("Upload log file (CSV)", type=["csv"])
    if uploaded_file:
        logs = pd.read_csv(uploaded_file)
        st.write("Log Data Preview:", logs.head())
        anomalies = detect_anomalies(logs)
        st.write("Detected Anomalies:", anomalies)
