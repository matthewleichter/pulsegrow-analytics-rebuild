import pandas as pd

def interpret_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interprets LLM output predictions and provides readable summaries.

    Expects DataFrame with a 'prediction' column.

    Returns a DataFrame with summary stats.
    """
    if 'prediction' not in df.columns:
        raise ValueError("DataFrame must contain 'prediction' column.")

    summary = df['prediction'].value_counts(normalize=True).reset_index()
    summary.columns = ['Prediction', 'Proportion']
    summary['Proportion'] = (summary['Proportion'] * 100).round(2)

    return summary
