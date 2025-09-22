# generate_all_datasets.py

import os
import pandas as pd
from utils.data_loader import generate_synthetic_users, generate_synthetic_transactions, generate_synthetic_usage_logs
from utils.feature_engineering import generate_churn_labels, generate_retention_data
from utils.ab_test_utils import generate_ab_test_results
from utils.causal_utils import generate_causal_treatments
from utils.segmentation_utils import generate_segmentation_labels
from utils.anomaly_utils import generate_anomaly_logs
from utils.funnel_utils import generate_funnel_steps
from utils.revenue_utils import generate_marketing_spend
from utils.preprocessing import ensure_directory

DATA_PATH = "data"

def generate_datasets():
    ensure_directory(DATA_PATH)

    print("[✓] Generating users.csv...")
    users_df = generate_synthetic_users()
    users_df.to_csv(os.path.join(DATA_PATH, "users.csv"), index=False)

    print("[✓] Generating transactions.csv...")
    tx_df = generate_synthetic_transactions(users_df)
    tx_df.to_csv(os.path.join(DATA_PATH, "transactions.csv"), index=False)

    print("[✓] Generating usage_logs.csv...")
    usage_df = generate_synthetic_usage_logs(users_df)
    usage_df.to_csv(os.path.join(DATA_PATH, "usage_logs.csv"), index=False)

    print("[✓] Generating churn_labels.csv...")
    churn_df = generate_churn_labels(usage_df)
    churn_df.to_csv(os.path.join(DATA_PATH, "churn_labels.csv"), index=False)

    print("[✓] Generating retention_data.csv...")
    retention_df = generate_retention_data(usage_df)
    retention_df.to_csv(os.path.join(DATA_PATH, "retention_data.csv"), index=False)

    print("[✓] Generating ab_test_results.csv...")
    ab_test_df = generate_ab_test_results(users_df)
    ab_test_df.to_csv(os.path.join(DATA_PATH, "ab_test_results.csv"), index=False)

    print("[✓] Generating causal_treatments.csv...")
    causal_df = generate_causal_treatments(users_df)
    causal_df.to_csv(os.path.join(DATA_PATH, "causal_treatments.csv"), index=False)

    print("[✓] Generating segmentation_labels.csv...")
    segmentation_df = generate_segmentation_labels(users_df)
    segmentation_df.to_csv(os.path.join(DATA_PATH, "segmentation_labels.csv"), index=False)

    print("[✓] Generating anomaly_logs.csv...")
    anomaly_df = generate_anomaly_logs(usage_df)
    anomaly_df.to_csv(os.path.join(DATA_PATH, "anomaly_logs.csv"), index=False)

    print("[✓] Generating funnel_steps.csv...")
    funnel_df = generate_funnel_steps(users_df)
    funnel_df.to_csv(os.path.join(DATA_PATH, "funnel_steps.csv"), index=False)

    print("[✓] Generating marketing_spend.csv...")
    marketing_df = generate_marketing_spend()
    marketing_df.to_csv(os.path.join(DATA_PATH, "marketing_spend.csv"), index=False)

    print("✅ All datasets generated successfully.")

if __name__ == "__main__":
    generate_datasets()
