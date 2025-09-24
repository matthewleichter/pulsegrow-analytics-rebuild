import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.segmentation_utils import perform_segmentation, plot_segment_clusters

def run_segmentation_model():
    st.title("Customer Segmentation")

    uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)

        try:
            clustered_df, labels = perform_segmentation(df, num_clusters=num_clusters)
            st.write("Clustered Data Preview:", clustered_df.head())

            fig = plot_segment_clusters(clustered_df, labels)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
