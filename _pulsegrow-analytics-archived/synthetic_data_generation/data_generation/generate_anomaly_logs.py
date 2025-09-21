
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
anomalies = []
for user in users['user_id']:
    if np.random.rand() < 0.1:
        anomalies.append({
            'user_id': user,
            'timestamp': pd.Timestamp('2022-02-01') + pd.to_timedelta(np.random.randint(0, 30), unit='D'),
            'anomaly_type': np.random.choice(['spike', 'drop', 'inactivity']),
            'severity': np.random.randint(1, 6)
        })
anomalies_df = pd.DataFrame(anomalies)
anomalies_df.to_csv('anomaly_logs.csv', index=False)
