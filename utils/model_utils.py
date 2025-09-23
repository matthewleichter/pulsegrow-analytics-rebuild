import joblib
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def run_marketing_mix_model(df: pd.DataFrame):
    """
    Runs a simple OLS model for Marketing Mix Modeling (MMM) and returns
    a summary DataFrame and a bar plot with confidence intervals.
    """
    if df.shape[1] < 2:
        return pd.DataFrame({"Error": ["Upload must include at least one target and one feature column."]}), None

    target_col = df.columns[0]
    feature_cols = df.columns[1:]

    X = df[feature_cols]
    X = sm.add_constant(X)
    y = df[target_col]

    model = sm.OLS(y, X).fit()

    # Coefficients and p-values
    summary_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.params,
        "P-Value": model.pvalues,
        "CI Lower": model.conf_int()[0],
        "CI Upper": model.conf_int()[1],
    }).reset_index(drop=True).round(4)

    # Plot with confidence intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        x=summary_df["Feature"],
        y=summary_df["Coefficient"],
        yerr=[
            summary_df["Coefficient"] - summary_df["CI Lower"],
            summary_df["CI Upper"] - summary_df["Coefficient"]
        ],
        fmt='o',
        color='blue',
        ecolor='gray',
        capsize=5
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("Marketing Channel Coefficients with 95% CI")
    ax.set_ylabel("Coefficient Value")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return summary_df, fig
