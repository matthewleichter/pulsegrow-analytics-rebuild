import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.funnel_utils import (
    generate_funnel_report,
    plot_funnel_chart,
    plot_conversion_rates
)

def run_funnel_analysis():
    st.title("🔄 Funnel Analysis")

    uploaded_file = st.file_uploader("Upload a CSV with 'user_id' and 'stage' columns", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            if 'user_id' not in data.columns or 'stage' not in data.columns:
                st.error("The CSV must contain both 'user_id' and 'stage' columns.")
                return

            st.subheader("Raw Funnel Data")
            st.dataframe(data.head())

            funnel_report = generate_funnel_report(data)

            st.subheader("📊 Funnel Conversion Report")
            st.dataframe(funnel_report)

            # Dual Plot Layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**User Drop-off per Stage**")
                fig1 = plot_funnel_chart(funnel_report)
                st.pyplot(fig1)

            with col2:
                st.markdown("**Conversion Rate by Stage**")
                fig2 = plot_conversion_rates(funnel_report)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred: {e}")
