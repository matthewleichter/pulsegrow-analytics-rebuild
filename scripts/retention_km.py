import pandas as pd
import streamlit as st
from utils.survival_utils import generate_retention_curve

def run_retention_km():
    st.title("User Retention - Kaplan Meier Curve")
    uploaded_file = st.file_uploader("Upload user retention data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data:", df.head())
        plot = generate_retention_curve(df)
        st.pyplot(plot)
