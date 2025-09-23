# utils/ab_test_utils.py
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from models.ab_testing import ABTestModel

def perform_ab_test(data: pd.DataFrame) -> dict:
    """
    Performs a two-sample t-test between control and treatment groups.

    Parameters:
        data (pd.DataFrame): A DataFrame with at least two columns:
            - 'group': categorical variable with values 'control' or 'treatment'
            - 'metric': numerical metric to compare (e.g. conversion rate, revenue)

    Returns:
        dict: Dictionary with means, standard deviations, p-value,
              confidence intervals, and effect size.
    """
    if 'group' not in data.columns or 'metric' not in data.columns:
        raise ValueError("Input data must contain 'group' and 'metric' columns")

    control = data[data['group'] == 'control']['metric'].dropna()
    treatment = data[data['group'] == 'treatment']['metric'].dropna()

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Not enough data in control or treatment group for t-test")

    # Calculate summary stats
    mean_control = control.mean()
    std_control = control.std()
    mean_treatment = treatment.mean()
    std_treatment = treatment.std()
    n_control = len(control)
    n_treatment = len(treatment)

    # Perform Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # Calculate 95% confidence interval for difference in means
    se_diff = np.sqrt(std_control**2 / n_control + std_treatment**2 / n_treatment)
    df = (std_control**2 / n_control + std_treatment**2 / n_treatment)**2 / (
        (std_control**2 / n_control)**2 / (n_control - 1) +
        (std_treatment**2 / n_treatment)**2 / (n_treatment - 1)
    )
    diff = mean_treatment - mean_control
    ci = stats.t.interval(0.95, df, loc=diff, scale=se_diff)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((std_control**2 + std_treatment**2) / 2)
    effect_size = diff / pooled_std

    return {
        'mean_control': round(mean_control, 4),
        'mean_treatment': round(mean_treatment, 4),
        'std_control': round(std_control, 4),
        'std_treatment': round(std_treatment, 4),
        'p_value': round(p_value, 4),
        'confidence_interval': (round(ci[0], 4), round(ci[1], 4)),
        'effect_size': round(effect_size, 4),
        'n_control': n_control,
        'n_treatment': n_treatment
    }

def compute_kl_divergence(data: pd.DataFrame) -> float:
    return ABTestModel.run_kl_divergence(data)

def visualize_ab_test(data: pd.DataFrame):
    st.subheader("Metric Distribution by Group")
    fig, ax = plt.subplots()
    sns.histplot(data=data, x="metric", hue="group", kde=True, stat="density", element="step")
    ax.set_title("Distribution of Metric by Group")
    st.pyplot(fig)

    st.subheader("T-Test Results")
    stats_result = perform_ab_test(data)
    for k, v in stats_result.items():
        st.markdown(f"**{k.replace('_', ' ').capitalize()}:** {v}")

    st.subheader("KL Divergence (Control || Treatment)")
    kl_value = compute_kl_divergence(data)
    st.markdown(f"**KL Divergence:** `{kl_value:.4f}`")
