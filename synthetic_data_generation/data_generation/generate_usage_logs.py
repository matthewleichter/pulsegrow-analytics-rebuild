
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
usage_logs = []
for user in users['user_id']:
    for _ in range(np.random.poisson(5)):
        usage_logs.append({
            'user_id': user,
            'timestamp': pd.Timestamp('2022-02-01') + pd.to_timedelta(np.random.randint(0, 60), unit='D'),
            'session_duration_min': np.random.exponential(30),
            'feature_used': np.random.choice(['dashboard', 'report', 'settings', 'chat', 'export'])
        })
usage_logs = pd.DataFrame(usage_logs)
usage_logs.to_csv('usage_logs.csv', index=False)
