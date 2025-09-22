import pandas as pd
import streamlit as st
from utils.causal_inference_utils import run_causal_model

def run_causal_inference():
    st.title("Causal Inference Module")
    uploaded_file = st.file_uploader("Upload treatment dataset (CSV)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        results = run_causal_model(data)
        st.write("Causal Inference Results:", results)
