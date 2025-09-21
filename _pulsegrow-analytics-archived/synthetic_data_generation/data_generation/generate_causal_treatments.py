
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
causal_treatments = pd.DataFrame({
    'user_id': users['user_id'],
    'treatment_group': np.random.choice(['control', 'treated'], size=len(users)),
    'engaged_after': np.random.choice([0, 1], size=len(users), p=[0.3, 0.7])
})
causal_treatments.to_csv('causal_treatments.csv', index=False)
