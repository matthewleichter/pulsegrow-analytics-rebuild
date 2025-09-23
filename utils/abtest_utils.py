import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from models.ab_testing import ABTestModel

def perform_ab_test(data):
    """
    Orchestrates A/B test visualizations and KL divergence computation.
    """

    st.subheader("A/B Test: Group Metric Distributions")
    show_group_distributions(data)

    st.subheader("KL Divergence Comparison")
    kl_value = ABTestModel.run_kl_divergence(data)

    st.markdown(f"**KL Divergence (Control || Treatment):** `{kl_value:.4f}`")

    st.info(
        "KL Divergence measures how one distribution diverges from another. "
        "A lower value suggests similar distributions, while a higher value implies stronger treatment effect."
    )


def show_group_distributions(data):
    """
    Visualize distributions of metrics across A/B groups.
    """
    fig, ax = plt.subplots()
    sns.histplot(data=data, x="metric", hue="group", kde=True, stat="density", common_norm=False, ax=ax)
    ax.set_title("Metric Distribution by Group")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Density")
    st.pyplot(fig)
