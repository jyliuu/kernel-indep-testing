import numpy as np

from .hsic import HSIC_test
from .kernel import gaussian_kernel_matrix


def simulate_dat(N=100, p=5, q=5, dependence=True):
    mean = np.zeros(p)
    A = np.random.randn(p, p)
    cov_matrix = A @ A.T
    X = np.random.multivariate_normal(mean, cov_matrix, N)

    if dependence:
        A = np.random.rand(p, q)  # generate some dependence
        Y = X @ A
    else:
        A = np.random.randn(q, q)
        cov_matrix = A @ A.T
        mean = np.zeros(q)
        Y = np.random.multivariate_normal(mean, cov_matrix, N)
    return X, Y


def simulate_dat2(N=100, p = 5, q=5, rho=0.5, mean=0):
    X_cov = np.eye(p)
    Y_cov = np.eye(q)

    XY_cov = np.full((p, q), rho)
    top_row = np.hstack((X_cov, XY_cov.T))  # Horizontal stack: (A, C^T)
    bottom_row = np.hstack((XY_cov, Y_cov))  # Horizontal stack: (C, B)

    cov = np.vstack((top_row, bottom_row))

    sim = np.random.multivariate_normal(mean, cov, N)
    return sim


def test_using_HSIC(X, Y, hsic_func = HSIC_test):
    K = gaussian_kernel_matrix(X, 2)
    L = gaussian_kernel_matrix(Y, 2)
    return hsic_func(K, L)