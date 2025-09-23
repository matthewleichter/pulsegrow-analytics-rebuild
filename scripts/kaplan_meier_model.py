import pandas as pd
import streamlit as st
from utils.kaplan_meier_utils import compute_km_curve

def run_kaplan_meier():
    st.title("Kaplan-Meier Retention Analysis")
    uploaded_file = st.file_uploader("Upload retention CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Check required columns
        if "retention_time" not in df.columns or "churned" not in df.columns:
            st.error("CSV must contain 'retention_time' and 'churned' columns.")
            return

        st.write("Data Preview:")
        st.dataframe(df.head())

        kmf = compute_km_curve(df)

        st.subheader("Kaplan-Meier Survival Function")
        st.line_chart(kmf.survival_function_)
