import streamlit as st
import pandas as pd
from utils.ab_test_utils import perform_ab_test
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization Function
def show_ab_test_visualizations(data):
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=data, x='group', y='metric', ax=ax1)
    ax1.set_title('Boxplot of Metric by Group')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(data=data, x='metric', hue='group', kde=True, element='step', stat='density')
    ax2.set_title('Distribution of Metric by Group')
    st.pyplot(fig2)

# Main Streamlit UI
def run_ab_test_analysis():
    st.title("A/B Test Analysis")

    uploaded_file = st.file_uploader("Upload CSV with 'group' and 'metric' columns", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data", data.head())

        try:
            result = perform_ab_test(data)
            st.subheader("Results")
            st.json(result)

            st.subheader("Visualizations")
            show_ab_test_visualizations(data)

        except Exception as e:
            st.error(f"Error: {e}")
