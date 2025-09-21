
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
funnel = pd.DataFrame({
    'user_id': users['user_id'],
    'step1_viewed': np.random.choice([0, 1], size=len(users), p=[0.1, 0.9]),
    'step2_clicked': np.random.choice([0, 1], size=len(users), p=[0.2, 0.8]),
    'step3_signed_up': np.random.choice([0, 1], size=len(users), p=[0.3, 0.7])
})
funnel.to_csv('funnel_steps.csv', index=False)
