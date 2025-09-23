import pandas as pd
from lifelines import KaplanMeierFitter

def compute_km_curve(data):
    """
    Computes Kaplan-Meier survival curve.

    Parameters:
        data (pd.DataFrame): Must contain 'retention_time' and 'churned' columns.

    Returns:
        KaplanMeierFitter object
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data["retention_time"], event_observed=data["churned"])
    return kmf
