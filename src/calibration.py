import numpy as np
from matplotlib import pyplot as plt

from src import simulate_dat2, test_using_HSIC


def calibration(B= 1000, P = 10000, N=100, p=2, q=2):
    # plot significance level against rejection rate under H0
    p_values = []
    for _ in range(B):
        X, Y = simulate_dat2(N, p, q, rho=0)
        T = test_using_HSIC(X, Y)
        permutation_res = np.array([test_using_HSIC(X, Y[np.random.permutation(np.arange(0, N)), :]) for _ in
                           range(P)])

        p_values.append(1-(permutation_res < T).mean())

    plt.hist(p_values, bins=40, edgecolor='black')  # `bins` specifies the number of intervals
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()