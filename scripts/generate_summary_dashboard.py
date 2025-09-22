# generate_summary_dashboard.py

import streamlit as st
from churn_model import run_churn_model
from segmentation import run_segmentation
from monetization_forecast import run_revenue_model
from funnel_analysis import run_funnel_analysis
from kaplan_meier_model import run_km_model
from llm_interpretation import generate_insights_from_llm

def generate_dashboard():
    st.set_page_config(page_title="PulseGrow Summary Dashboard", layout="wide")

    st.title("📊 PulseGrow Analytics: Executive Summary")

    with st.spinner("Running Churn Model..."):
        churn_results = run_churn_model()
        st.subheader("🔁 Churn Prediction")
        st.write(churn_results)

    with st.spinner("Running Segmentation Model..."):
        segmentation_results = run_segmentation()
        st.subheader("🧬 Customer Segmentation")
        st.write(segmentation_results)

    with st.spinner("Running Revenue Forecast..."):
        revenue_results = run_revenue_model()
        st.subheader("💰 Monetization Forecast")
        st.write(revenue_results)

    with st.spinner("Running Funnel Analysis..."):
        funnel_fig = run_funnel_analysis()
        st.subheader("🛒 Funnel Drop-off")
        st.plotly_chart(funnel_fig, use_container_width=True)

    with st.spinner("Running Kaplan-Meier Survival Model..."):
        km_fig = run_km_model()
        st.subheader("📈 Retention Survival Curve")
        st.plotly_chart(km_fig, use_container_width=True)

    with st.spinner("Generating LLM Interpretation..."):
        llm_insights = generate_insights_from_llm({
            "churn": churn_results,
            "segmentation": segmentation_results,
            "revenue": revenue_results
        })
        st.subheader("🧠 AI-Powered Strategic Insights")
        st.markdown(llm_insights)

if __name__ == "__main__":
    generate_dashboard()
