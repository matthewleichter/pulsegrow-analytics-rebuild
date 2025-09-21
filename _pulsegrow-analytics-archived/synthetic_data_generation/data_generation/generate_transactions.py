
import pandas as pd
import numpy as np

users = pd.read_csv('users.csv')
n_tx = 3000
transactions = pd.DataFrame({
    'transaction_id': [f'tx_{i}' for i in range(n_tx)],
    'user_id': np.random.choice(users['user_id'], size=n_tx),
    'product_id': [f'prod_{i % 100}' for i in range(n_tx)],
    'purchase_date': pd.date_range('2022-02-01', periods=n_tx, freq='H'),
    'amount': np.random.gamma(shape=2.0, scale=20.0, size=n_tx).round(2)
})
transactions.to_csv('transactions.csv', index=False)
