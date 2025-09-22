import pandas as pd
import streamlit as st
from utils.funnel_utils import generate_funnel_report

def run_funnel_analysis():
    st.title("Funnel Analysis")
    uploaded_file = st.file_uploader("Upload funnel steps CSV", type=["csv"])
    if uploaded_file:
        funnel_data = pd.read_csv(uploaded_file)
        st.write("Funnel Data Preview:", funnel_data.head())
        report = generate_funnel_report(funnel_data)
        st.write("Funnel Report:", report)
