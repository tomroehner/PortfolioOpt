import numpy as np
from sklearn.cluster import KMeans
from random import randrange
import copy


def standardize(ar: np.ndarray):
    mean_cols = np.mean(ar, axis=0)
    mean_cols_m = np.tile(mean_cols, (ar.shape[0], 1))
    sd_cols = np.std(ar, axis=0)
    sd_cols_m = np.tile(sd_cols, (ar.shape[0], 1))
    ar = (ar - mean_cols_m) / sd_cols_m
    return ar


def prepare_pars(mean_returns: np.ndarray, cov_returns: np.ndarray):
    mean_returns = mean_returns.reshape(len(mean_returns), 1)
    asset_pars = np.concatenate([mean_returns, cov_returns], axis=1)
    return standardize(asset_pars)


def elbow_curve(mean_returns: np.ndarray, cov_returns: np.ndarray, min_clusters: int, max_clusters: int,
                n_runs: int = 10):
    """
    Runs the k-means clustering algorithm repeatedly for a specified range of numbers of clusters. Returns a list of
    the sum of the squared error (SSE) of each run.

    :param mean_returns:
        Mean returns for the individual stocks
    :param cov_returns:
        Covariance matrix of the stock returns
    :param min_clusters:
        Minimum amount of clusters the algorithm should be run with
    :param max_clusters:
        Maximum amount of clusters the algorithm should be run with
    :param n_runs:
        Number of times the k-means algorithm is run for every value of k
    :return:
        List of the sum of the squared error (SSE) of each run
    """
    asset_pars = prepare_pars(mean_returns, cov_returns)
    sse = []
    for n_clusters in range(min_clusters, max_clusters):
        assets_cluster = KMeans(algorithm='lloyd', n_clusters=n_clusters, n_init=n_runs)
        assets_cluster.fit(asset_pars)
        sse.append(assets_cluster.inertia_)
    return sse


def form_clusters(mean_returns: np.ndarray, cov_returns: np.ndarray, n_clusters: int = 10, n_runs: int = 10):
    """
    Applies the k-Means-Algorithm to form clusters based on the historical market behaviour of the stocks

    :param mean_returns:
        Mean returns for the individual stocks
    :param cov_returns:
        Covariance matrix of the stock returns
    :param n_clusters:
        Number of clusters to form
    :param n_runs:
        Number of times the k-means algorithm is run
    :return:
        The sklearn.cluster.KMeans object containing the clustering result
    """
    pars = prepare_pars(mean_returns, cov_returns)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_runs)
    kmeans.fit(pars)
    return kmeans


def clusters_from_labels(labels, ticker_symbols, n_clusters):
    clusters = [[] for _ in range(n_clusters)]
    for i in range(len(labels)):
        clusters[labels[i]].append(ticker_symbols[i])
    return clusters


def pick_random(stock_clusters: list[list], n=None) -> list[str]:
    """
    Picks n random stocks evenly from a list of clusters.

    :param stock_clusters:
        List of clusters
    :param n:
        Amount of stocks to be picked
    :return:
        List of selected stocks given by their ticker symbol
    """
    portfolio = []
    if n is None:
        for cluster in stock_clusters:
            r = randrange(len(cluster))
            portfolio.append(cluster[r])
    else:
        _stock_clusters = copy.deepcopy(stock_clusters)
        while len(portfolio) < n:
            open_clusters = [x for x in range(len(_stock_clusters))]
            while len(open_clusters) > 0 and len(portfolio) < n:
                # pick random cluster
                r_c = randrange(len(open_clusters))
                c_pick = open_clusters[r_c]
                # remove selected cluster from selection
                open_clusters.pop(r_c)
                # pick random stock from selected cluster
                r_s = randrange(len(_stock_clusters[c_pick]))
                # add selected stock to portfolio
                portfolio.append(_stock_clusters[c_pick][r_s])
                # remove selected stock from cluster
                _stock_clusters[c_pick].pop(r_s)
                if len(_stock_clusters[c_pick]) == 0:
                    _stock_clusters.pop(c_pick)
                    for i in range(len(open_clusters)):
                        if open_clusters[i] > c_pick:
                            open_clusters[i] = open_clusters[i] - 1

    return portfolio
