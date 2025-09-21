
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def run_kaplan_meier_model(data_path="data/retention_data.csv", output_path="assets/kaplan_meier_curve.png"):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["retention_time", "churned"])

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["retention_time"], event_observed=df["churned"])

    ax = kmf.plot(ci_show=True)
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    plt.savefig(output_path)
    plt.close()
