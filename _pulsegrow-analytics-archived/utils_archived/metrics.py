from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

def evaluate_classification(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }

def evaluate_regression(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }