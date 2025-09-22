
# PulseGrow Synthetic Dataset Generator

This folder contains the complete system for generating the synthetic datasets required for the PulseGrow analytics platform. These datasets are all linked through common identifiers like `user_id` and `event_id`, and were generated to simulate real-world analytics scenarios in SaaS platforms.

## 📁 Included Datasets

| Filename                  | Description |
|---------------------------|-------------|
| users.csv                 | Base user demographics and signup data |
| transactions.csv          | Purchase transactions per user |
| usage_logs.csv            | App usage logs (daily level) |
| retention_data.csv        | Retention records by cohort |
| churn_labels.csv          | Binary churn classification (0 or 1) |
| events.csv                | Clickstream-style event logs |
| ab_test_results.csv       | Synthetic A/B testing output |
| anomaly_logs.csv          | Anomaly detection events |
| causal_treatments.csv     | Users marked as treated/untreated |
| funnel_steps.csv          | Funnel progress (signup → conversion) |
| marketing_spend.csv       | Weekly marketing channel spend |
| product_features.csv      | Features of various product SKUs |
| segmentation_labels.csv   | Clustered user segments |

## ⚙️ How to Use

To generate all datasets, run:

```bash
python main.py
```

Ensure all `generate_*.py` scripts are in the same folder. This will populate the `data/` directory with 13 linked datasets.

## 📊 Preview: users.csv

![users.csv screenshot](users_csv_preview.png)

---

Each dataset is time-stamped and synchronized across keys like `user_id`, `timestamp`, or `event_id` to ensure analytical consistency.

For support or customization, please contact the PulseGrow team.
