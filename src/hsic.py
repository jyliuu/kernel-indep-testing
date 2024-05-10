import numpy as np


def HSIC_test(K, L):
    m = K.shape[0]
    return np.sum(K * L) / m ** 2 + np.sum(K) * np.sum(L) / m ** 4 - np.einsum('ij,ik->', K, L) * 2 / m ** 3


def HSIC_test2(K, L):
    I = np.eye(K.shape[0])
    ones = np.ones(K.shape)
    H = I - ones/K.shape[0]

    return np.trace(K @ H @ L @ H)/K.shape[0]**2