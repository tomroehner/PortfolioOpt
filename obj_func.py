import numpy as np


def max_sharpe_ratio(weights, mean_returns, cov_returns, risk_free_rate):
    numerator = np.matmul(mean_returns, weights.T) - risk_free_rate
    denominator = np.sqrt(np.matmul(np.matmul(weights, cov_returns), weights.T))
    return -1 * (numerator / denominator)


def max_expected_return(weights, mean_returns):
    return -1 * np.matmul(mean_returns, weights.T)


def min_variance(weights, cov_returns):
    return np.matmul(np.matmul(weights, cov_returns), weights.T)
