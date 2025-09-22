import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.ab_test_utils import perform_ab_test

def run_ab_test_analysis():
    st.title("A/B Test Analysis")
    uploaded_file = st.file_uploader("Upload A/B test results CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        result = perform_ab_test(data)
        st.write("A/B Test Result:", result)
