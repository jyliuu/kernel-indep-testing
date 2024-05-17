import numpy as np

from src import simulate_dat2, test_using_HSIC_, gaussian_kernel_matrix, test_using_HSIC


def get_p_values_vs_uniform(
    kernel,
    B=100,
    P=100,
    N=100,
    p=2, q=3
):
    # plot significance level against rejection rate under H0
    p_values = []
    sigmas = []
    for _ in range(B):
        X, Y = simulate_dat2(N, p, q, rho=0)
        T, sigma = test_using_HSIC_(X, Y, kernel=kernel)
        permutation_res = np.array([test_using_HSIC(X, Y[np.random.permutation(np.arange(0, N)), :], kernel=kernel) for _ in
                           range(P)])

        p_val = 1-(permutation_res < T).mean()
        p_values.append(p_val)
        sigmas.append(sigma)

    return p_values, sigmas

def calibration():
    res = []

    for sigma in np.arange(0.5, 1.8, 0.1):
        print(f'sigma: {sigma}')
        p_values, _ = get_p_values_vs_uniform(kernel=lambda Z: gaussian_kernel_matrix(Z, sigma=sigma))
        res.append((sigma, p_values))
        print(p_values)

    return res