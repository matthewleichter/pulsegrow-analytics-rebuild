
import streamlit as st

def generate_dashboard(metrics_dict):
    st.title("PulseGrow Analytics Dashboard")
    for section, metrics in metrics_dict.items():
        st.subheader(section)
        for key, value in metrics.items():
            st.write(f"{key}: {value}")
