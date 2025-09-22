
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
ab_test = pd.DataFrame({
    'user_id': users['user_id'],
    'variant': np.random.choice(['A', 'B'], size=len(users)),
    'converted': np.random.choice([0, 1], size=len(users), p=[0.6, 0.4])
})
ab_test.to_csv('ab_test_results.csv', index=False)
