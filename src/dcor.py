import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances


def V_n(X: np.ndarray, Y: np.ndarray) -> float:
    dists_X = pairwise_distances(X, metric="sqeuclidean")
    dists_Y = pairwise_distances(Y, metric="sqeuclidean")

    n = X.shape[0]
    V_n = (
        np.sum(dists_X * dists_Y) / n**2
        + np.sum(dists_X) * np.sum(dists_Y) / n**4
        - 2 * np.einsum("ij,ik->", dists_X, dists_Y) / n**3
    )
    return V_n


def dCor1(X: np.ndarray, Y: np.ndarray) -> float:
    XY_V_n = V_n(X, Y)
    X_V_n = V_n(X, X)
    Y_V_n = V_n(Y, Y)

    R_n = XY_V_n / (np.sqrt(X_V_n * Y_V_n))
    return R_n


def center_dists(X: np.ndarray) -> np.ndarray:
    X_dists = distance_matrix(X, X)

    X_colmeans = np.mean(X_dists, axis=0)
    X_rowmeans = np.mean(X_dists, axis=1, keepdims=True)
    X_mean = np.mean(X_dists)

    A = X_dists - X_colmeans - X_rowmeans + X_mean

    return A


def V_n2(X: np.ndarray, Y: np.ndarray) -> float:
    X_centered = center_dists(X)
    Y_centered = center_dists(Y)
    return np.mean(X_centered * Y_centered)


def dCor2(X: np.ndarray, Y: np.ndarray) -> float:
    XY_V_n = V_n2(X, Y)
    X_V_n = V_n2(X, X)
    Y_V_n = V_n2(Y, Y)

    R_n = XY_V_n / (np.sqrt(X_V_n * Y_V_n))
    return R_n


def test_using_dCor(X: np.ndarray, Y: np.ndarray) -> float:
    return X.shape[0] * V_n2(X, Y)
