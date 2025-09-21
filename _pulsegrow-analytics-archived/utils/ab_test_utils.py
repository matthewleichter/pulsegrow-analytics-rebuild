import numpy as np
import scipy.stats as stats

def bootstrap_ab_test(control, test, num_bootstrap=1000):
    diffs = []
    for _ in range(num_bootstrap):
        c_sample = np.random.choice(control, size=len(control), replace=True)
        t_sample = np.random.choice(test, size=len(test), replace=True)
        diffs.append(t_sample.mean() - c_sample.mean())
    return np.percentile(diffs, [2.5, 97.5])