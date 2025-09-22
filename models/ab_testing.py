import scipy.stats as stats

def ab_test(group_a, group_b):
    t_stat, p_val = stats.ttest_ind(group_a, group_b)
    return t_stat, p_val