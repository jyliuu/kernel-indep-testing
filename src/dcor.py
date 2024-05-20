import numpy as np
from sklearn.metrics import pairwise_distances


def V_n(X, Y):
    dists_X = pairwise_distances(X, metric='sqeuclidean')
    dists_Y = pairwise_distances(Y, metric='sqeuclidean')

    n = X.shape[0]
    V_n = np.sum(dists_X * dists_Y) / n ** 2 + np.sum(dists_X) * np.sum(dists_Y) / n ** 4 - np.einsum('ij,ik->', dists_X, dists_Y) * 2 / n ** 3
    return V_n


def test_using_dCor(X, Y):
    XY_V_n = V_n(X, Y)
    X_V_n = V_n(X, X)
    Y_V_n = V_n(Y, Y)

    R_n = XY_V_n/(np.sqrt(X_V_n) * np.sqrt(Y_V_n))
    return R_n
