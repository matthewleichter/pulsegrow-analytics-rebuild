import pandas as pd
import streamlit as st
from utils.model_utils import run_marketing_mix_model

def run_marketing_mix():
    st.title("Marketing Mix Modeling")
    uploaded_file = st.file_uploader("Upload marketing spend CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())

        model_output, fig = run_marketing_mix_model(data)

        st.write("Model Output:", model_output)

        if fig:
            st.pyplot(fig)
