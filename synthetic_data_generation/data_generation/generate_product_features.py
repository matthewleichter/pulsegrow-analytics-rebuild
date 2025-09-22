
import pandas as pd
import numpy as np

features = ['dashboard', 'chat', 'analytics', 'settings', 'export']
product_features = pd.DataFrame({
    'feature_name': features,
    'popularity_score': np.random.uniform(0, 1, len(features)),
    'avg_session_duration_min': np.random.normal(15, 5, len(features)).clip(min=0)
})
product_features.to_csv('product_features.csv', index=False)
