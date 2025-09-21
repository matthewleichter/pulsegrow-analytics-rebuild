import pandas as pd
from lifelines import KaplanMeierFitter

def compute_survival(data):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data['tenure'], event_observed=data['churned'])
    return kmf