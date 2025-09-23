import pandas as pd
import streamlit as st
from utils.model_utils import run_marketing_mix_model

def run_marketing_mix():
    st.title("📊 Marketing Mix Modeling")

    uploaded_file = st.file_uploader("Upload marketing spend CSV (target column first, features next)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(data.head())

        summary_df, fig = run_marketing_mix_model(data)

        st.subheader("📈 Model Summary Table")
        st.dataframe(summary_df)

        if fig:
            st.subheader("📉 Coefficient Impact with 95% Confidence Intervals")
            st.pyplot(fig)
