import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter
from utils.kaplan_meier_utils import compute_km_curve

def run_kaplan_meier():
    st.title("Kaplan-Meier Retention Analysis")
    uploaded_file = st.file_uploader("Upload retention CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        kmf = compute_km_curve(df)
        st.write("Survival Function:")
        st.line_chart(kmf.survival_function_)
