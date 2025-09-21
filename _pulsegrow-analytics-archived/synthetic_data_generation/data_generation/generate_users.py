
import pandas as pd
import numpy as np

np.random.seed(42)
n_users = 1000
users = pd.DataFrame({
    'user_id': [f'user_{i}' for i in range(n_users)],
    'signup_date': pd.date_range('2022-01-01', periods=n_users, freq='D'),
    'age': np.random.randint(18, 65, size=n_users),
    'gender': np.random.choice(['Male', 'Female', 'Other'], size=n_users),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=n_users)
})
users.to_csv('users.csv', index=False)
