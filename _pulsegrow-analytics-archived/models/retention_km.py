from lifelines import KaplanMeierFitter

def plot_kaplan_meier(df):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["duration"], event_observed=df["churned"])
    return kmf