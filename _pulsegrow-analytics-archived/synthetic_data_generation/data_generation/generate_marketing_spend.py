
import pandas as pd
import numpy as np

channels = ['email', 'social', 'search', 'affiliate']
marketing = pd.DataFrame({
    'channel': np.random.choice(channels, size=60),
    'month': pd.date_range(start='2022-01-01', periods=60, freq='M'),
    'spend_usd': np.random.gamma(2000, 2, size=60)
})
marketing.to_csv('marketing_spend.csv', index=False)
