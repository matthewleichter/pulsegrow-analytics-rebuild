import streamlit as st
import pandas as pd
from utils.ab_test_utils import perform_ab_test

def run_ab_test_analysis():
    st.title("A/B Test Analysis")

    uploaded_file = st.file_uploader("Upload CSV with 'group' and 'metric' columns", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data", data.head())

        try:
            result = perform_ab_test(data)
            st.subheader("Results")
            st.json(result)
        except Exception as e:
            st.error(f"Error: {e}")
