import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.segmentation_utils import (
    perform_segmentation,
    plot_segmentation,
    load_segmentation_results
)

def run_segmentation_model():
    st.title("Customer Segmentation")
    st.markdown("Upload your customer dataset or use our default example to segment users into clusters using KMeans and visualize with PCA.")

    uploaded_file = st.file_uploader("📁 Upload customer data CSV", type=["csv"])

    # Load user-uploaded or default data
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info("✅ Using uploaded file.")
    else:
        default_path = "data/sample_segmentation_data.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.info("ℹ️ Using default sample data from `data/sample_segmentation_data.csv`.")
        else:
            st.warning("⚠️ No uploaded file and default example not found.")
            return

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Cluster slider
    num_clusters = st.slider("🔢 Select number of clusters", min_value=2, max_value=10, value=4)

    try:
        # Run clustering
        clustered_df, labels = perform_segmentation(df, num_clusters=num_clusters)
        st.subheader("🧠 Clustered Data")
        st.dataframe(clustered_df.head())

        # Cluster plot
        fig = plot_segmentation(clustered_df, labels)
        st.subheader("📊 Cluster Visualization (PCA)")
        st.pyplot(fig)

        # Summary
        results_df = load_segmentation_results(clustered_df)
        st.subheader("📈 Cluster Summary")
        st.dataframe(results_df)

    except Exception as e:
        st.error(f"Segmentation failed: {e}")
