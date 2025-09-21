
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
churn = pd.DataFrame({
    'user_id': users['user_id'],
    'churned': np.random.choice([0, 1], size=len(users), p=[0.7, 0.3])
})
churn.to_csv('churn_labels.csv', index=False)
