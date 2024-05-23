from concurrent.futures import ProcessPoolExecutor

import numpy as np
from dcor.independence import distance_covariance_test

from src import simulate_dat2, test_using_HSIC_, gaussian_kernel_matrix, test_using_HSIC
from src.dcor import test_using_dCor


def permutation_test(X, Y, test_method, P):
    N = len(X)
    T, other_var = test_method(X, Y)
    permutation_res = np.array(
        [test_method(X, Y[np.random.permutation(np.arange(0, N)), :])[0] for _ in range(P)]
    )

    p_val = 1 - (permutation_res < T).mean()
    return p_val, T, other_var, permutation_res

def simulate_p_values_resampling_from_test(
        test_method,
        simulate_dat,
        B=100,
        P=100
):
    p_vals = []
    other_vars = []
    for i in range(B):
        print(f'Iteration: {i}')
        X, Y = simulate_dat()

        p_val, T, other_var, permutation_res = permutation_test(X, Y, test_method, P)

        p_vals.append(p_val)
        other_vars.append(other_var)

    return p_vals, other_vars


def simulate_p_values_resampling_from_HSIC(N=100, B=100, P=100):
    return simulate_p_values_resampling_from_test(
        test_method=lambda X, Y: test_using_HSIC_(X, Y),
        simulate_dat=lambda: simulate_dat2(N),
        B=B, P=P
    )


def simulate_p_values_resampling_from_dCor(N=100, B=100, P=100):
    return simulate_p_values_resampling_from_test(
        test_method=lambda X, Y: (test_using_dCor(X, Y), None),
        simulate_dat=lambda: simulate_dat2(N),
        B=B, P=P
    )


def simulate_p_values_resampling_from_dCor2(simulate_dat, B=100, P=100):
    p_vals = []
    for i in range(B):
        print('Iteration:', i)
        X, Y = simulate_dat()
        res = distance_covariance_test(X, Y, num_resamples=P)
        p_vals.append(res.pvalue)
    return p_vals

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
        permutation_res = np.array(
            [test_using_HSIC(X, Y[np.random.permutation(np.arange(0, N)), :], kernel=kernel) for _ in
             range(P)])

        p_val = 1 - (permutation_res < T).mean()
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


def get_rho_steps(start=0.5, end=0.001, n_steps=20):
    steps = np.logspace(np.log10(start), np.log10(end), num=n_steps)
    return steps


def sim_rho_going_to_0(sim_p_vals, get_rho_steps=get_rho_steps):
    steps = get_rho_steps()

    with ProcessPoolExecutor() as executor:
        results = np.array(list(executor.map(sim_p_vals, steps)))

    means = results.mean(axis=1)
    rejection_rates = (results < 0.05).mean(axis=1)
    return steps, means, rejection_rates
