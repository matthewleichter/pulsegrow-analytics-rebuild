
from generate_users import generate_users
from generate_transactions import generate_transactions
from generate_usage_logs import generate_usage_logs
from generate_retention_data import generate_retention_data
from generate_churn_labels import generate_churn_labels
from generate_events import generate_events
from generate_ab_test_results import generate_ab_test_results
from generate_anomaly_logs import generate_anomaly_logs
from generate_causal_treatments import generate_causal_treatments
from generate_funnel_steps import generate_funnel_steps
from generate_marketing_spend import generate_marketing_spend
from generate_product_features import generate_product_features
from generate_segmentation_labels import generate_segmentation_labels

def run_all_generators():
    print("Generating users.csv...")
    generate_users()

    print("Generating transactions.csv...")
    generate_transactions()

    print("Generating usage_logs.csv...")
    generate_usage_logs()

    print("Generating retention_data.csv...")
    generate_retention_data()

    print("Generating churn_labels.csv...")
    generate_churn_labels()

    print("Generating events.csv...")
    generate_events()

    print("Generating ab_test_results.csv...")
    generate_ab_test_results()

    print("Generating anomaly_logs.csv...")
    generate_anomaly_logs()

    print("Generating causal_treatments.csv...")
    generate_causal_treatments()

    print("Generating funnel_steps.csv...")
    generate_funnel_steps()

    print("Generating marketing_spend.csv...")
    generate_marketing_spend()

    print("Generating product_features.csv...")
    generate_product_features()

    print("Generating segmentation_labels.csv...")
    generate_segmentation_labels()

    print("✅ All synthetic datasets generated successfully.")

if __name__ == "__main__":
    run_all_generators()
