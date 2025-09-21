
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
segments = ['power_user', 'casual_user', 'infrequent_user']
segmentation = pd.DataFrame({
    'user_id': users['user_id'],
    'segment': np.random.choice(segments, size=len(users), p=[0.2, 0.5, 0.3])
})
segmentation.to_csv('segmentation_labels.csv', index=False)
