# models/ab_test.py

import numpy as np
from scipy.stats import entropy

class ABTestModel:
    def __init__(self, data):
        self.data = data
        self.group_a = data[data['group'] == 'A']['metric']
        self.group_b = data[data['group'] == 'B']['metric']

    def run_kl_divergence(self, bins=50):
        # Calculate histograms for both groups
        hist_a, bin_edges = np.histogram(self.group_a, bins=bins, density=True)
        hist_b, _ = np.histogram(self.group_b, bins=bin_edges, density=True)

        # Add a small constant to avoid division by zero
        hist_a += 1e-10
        hist_b += 1e-10

        kl_ab = entropy(hist_a, hist_b)
        kl_ba = entropy(hist_b, hist_a)

        return {
            'kl_ab': kl_ab,
            'kl_ba': kl_ba,
            'bin_edges': bin_edges,
            'hist_a': hist_a,
            'hist_b': hist_b
        }


#import scipy.stats as stats
#
#def ab_test(group_a, group_b):
#    t_stat, p_val = stats.ttest_ind(group_a, group_b)
#    return t_stat, p_val
