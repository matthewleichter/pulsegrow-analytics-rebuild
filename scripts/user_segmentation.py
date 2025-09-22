
import pandas as pd

def segment_users_by_behavior(data: pd.DataFrame) -> pd.DataFrame:
    if 'usage_count' not in data.columns:
        raise ValueError("Expected column 'usage_count' not found in input data.")
    data['segment'] = pd.qcut(data['usage_count'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    return data
