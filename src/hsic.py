import numpy as np

from .kernel import gaussian_kernel_matrix


def HSIC_test(K, L):
    m = K.shape[0]
    return np.sum(K * L) / m ** 2 + np.sum(K) * np.sum(L) / m ** 4 - np.einsum('ij,ik->', K, L) * 2 / m ** 3


def HSIC_test2(K, L):
    I = np.eye(K.shape[0])
    ones = np.ones(K.shape)
    H = I - ones/K.shape[0]

    return np.trace(K @ H @ L @ H)/K.shape[0]**2


def test_using_HSIC_(X, Y, hsic_func=HSIC_test, kernel=gaussian_kernel_matrix):
    K, sigma_X = kernel(X)
    L, sigma_Y = kernel(Y)
    return hsic_func(K, L), (sigma_X, sigma_Y)


def test_using_HSIC(*args, **kwargs):
    hsic_T, sigmas = test_using_HSIC_(*args, **kwargs)
    return hsic_T
