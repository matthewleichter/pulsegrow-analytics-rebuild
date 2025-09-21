
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
retention = pd.DataFrame({
    'user_id': users['user_id'],
    'retention_days': np.random.exponential(50, size=len(users)).astype(int)
})
retention.to_csv('retention_data.csv', index=False)
