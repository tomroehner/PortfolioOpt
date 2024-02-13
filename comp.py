import pandas as pd
import numpy as np


def compute_stock_returns(close_prices: pd.DataFrame) -> np.ndarray:
    """
    Computes the returns for a given dataset of stock prices

    :param close_prices:
        Close prices of a list of individual stocks
    :return:
        Returns of the individual stocks
    """
    rows = close_prices.shape[0]
    cols = close_prices.shape[1]
    stock_returns = np.zeros([rows - 1, cols])
    for c in range(cols):
        for r in range(rows - 1):
            stock_returns[r, c] = ((close_prices.iloc[r + 1, c] - close_prices.iloc[r, c]) /
                                   close_prices.iloc[r, c])

    return stock_returns


def exp_pf_return(mean_returns: np.ndarray, weights: np.ndarray):
    """
    Calculates the expected return for a given portfolio

    :param mean_returns:
        The mean of the returns of each stock of the portfolio
    :param weights:
        The weights associated with each stock of the portfolio
    :return:
        The expected return of the portfolio
    """
    res = np.matmul(mean_returns, weights.T)
    return res


def pf_risk(weights: np.ndarray, cov_returns: np.ndarray):
    """
    Calculates the standard deviation of the returns (risk) of the portfolio

    :param weights:
        The weights associated with each stock of the portfolio
    :param cov_returns:
        The covariance matrix of the returns of a given portfolio
    :return:
        The portfolio risk
    """
    s1 = np.matmul(weights, cov_returns)
    s2 = np.matmul(s1, weights.T)
    return np.sqrt(s2)


def pf_sharpe_ratio(ann_return, ann_risk, ann_risk_free_rate):
    """
    Calculates the annual Sharpe Ratio

    :param ann_return:
        Annual return
    :param ann_risk:
        Annual risk
    :param ann_risk_free_rate:
        Annual risk-free rate
    :return:
        Sharpe Ratio
    """
    return (ann_return - ann_risk_free_rate) / ann_risk


def div_ratio(weights, cov_returns):
    """
    Calculates the diversification ratio of a portfolio

    :param weights:
        The portfolio weights
    :param cov_returns:
        Covariance matrix of the stock returns
    :return:
        Diversification ratio
    """
    var_r = np.diagonal(cov_returns)
    sd_r = np.sqrt(var_r)
    d = np.matmul(weights, sd_r)
    pf_sd = np.sqrt(np.matmul(np.matmul(weights, cov_returns), weights.T))
    return d / pf_sd
