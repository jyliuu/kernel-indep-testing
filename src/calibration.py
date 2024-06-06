from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Tuple

import numpy as np
from dcor.independence import distance_covariance_test
from joblib import Parallel, delayed

from src import simulate_dat2, test_using_HSIC_, gaussian_kernel_matrix, test_using_HSIC


def get_power_under(
    p_vals_methods,
    dims=[2, 10, 100, 150],
    rhos=[0.1, 0.05, 0.01, 0.006],
    add_sigma_scaled=False,
    N=100,
    B=500,
):
    final_res = []

    for i, (d, rho) in enumerate(zip(dims, rhos)):
        # Plot for this particular combination
        print("Testing dimension", d, "with rho", rho)

        methods = p_vals_methods

        if add_sigma_scaled:
            sigma_scaled_wrt_dim = {
                rf"$\sigma={s:.2f}$": partial(
                    permutation_test_p_val,
                    test_method=partial(
                        test_using_HSIC, kernel=partial(gaussian_kernel_matrix, sigma=s)
                    ),
                    P=299,
                )
                for s in np.power(d, d[0.25, 0.5, 0.75, 1])
            }

            methods.update(sigma_scaled_wrt_dim)

        res_for_conf = []
        for name, method in methods.items():
            print("Testing", name)
            p_vals = get_p_vals_B_times(
                method,
                partial(simulate_dat2, N=N, rho=rho, p=d, q=d),
                B=B,
                parallel=True,
            )
            res_for_conf.append(p_vals)
        final_res.append(res_for_conf)

    return final_res


def permutation_test_p_val(
    X: np.ndarray,
    Y: np.ndarray,
    test_method: Callable[[np.ndarray, np.ndarray], float],
    P: int,
) -> float:
    N = len(X)
    T = test_method(X, Y)
    permutation_res = np.array(
        [test_method(X, Y[np.random.permutation(N), :]) for _ in range(P)]
    )

    extremes = (permutation_res > T).sum()

    return (1 + extremes) / (P + 1)


def get_p_vals_B_times(
    test_method: Callable[[np.ndarray, np.ndarray], float],
    simulate_dat: Callable[[], Tuple[np.ndarray, np.ndarray]],
    B: int = 100,
    seed=14,
    parallel=False,
) -> np.ndarray:

    def parallelizable(i):
        print(f"Iteration: {i}")
        np.random.seed(seed * i)
        X, Y = simulate_dat()
        p_val = test_method(X, Y)
        return p_val

    if parallel:
        p_vals = Parallel(n_jobs=-1)(delayed(parallelizable)(i) for i in range(B))
    else:
        p_vals = [parallelizable(i) for i in range(B)]

    return np.array(p_vals)


def simulate_p_values_resampling_from_HSIC(simulate_dat, B=100, P=100):
    return get_p_vals_B_times(
        test_method=lambda X, Y: test_using_HSIC_(
            X, Y, kernel=lambda Z: gaussian_kernel_matrix(Z, 100)
        ),
        simulate_dat=simulate_dat,
        B=B,
        P=P,
    )


def simulate_p_values_resampling_from_dCor(simulate_dat, B=100, P=100):
    p_vals = []
    for i in range(B):
        print("Iteration:", i)
        X, Y = simulate_dat()
        res = distance_covariance_test(X, Y, num_resamples=P)
        p_vals.append(res.pvalue)
    return p_vals


def d_cov_test_p_val(X, Y, P=100):
    res = distance_covariance_test(X, Y, num_resamples=P)
    return res.pvalue


def get_p_values_vs_uniform(kernel, B=100, P=100, N=100, p=2, q=2, rho=0):
    # plot significance level against rejection rate under H0
    p_values = []
    for i in range(B):
        np.random.seed(i * 69)
        X, Y = simulate_dat2(N, p, q, rho)
        p_val = permutation_test_p_val(
            X,
            Y,
            test_method=partial(test_using_HSIC, kernel=kernel),
            P=P,
        )
        p_values.append(p_val)

    return p_values


def get_p_values_from_sigma(sigma, **kwargs):
    print(f"sigma: {sigma or 0:.2f}")
    p_values = get_p_values_vs_uniform(
        kernel=partial(gaussian_kernel_matrix, sigma=sigma), **kwargs
    )
    return sigma, p_values


def calibration(sigma_start=0.1, sigma_end=1, step=0.1, parallel=False, **kwargs):
    partial_get_p_values_from_sigma = partial(get_p_values_from_sigma, **kwargs)

    if parallel:
        res = Parallel(n_jobs=-1)(
            delayed(partial_get_p_values_from_sigma)(sigma)
            for sigma in [*np.arange(sigma_start, sigma_end, step), None]
        )
    else:
        res = [
            partial_get_p_values_from_sigma(sigma)
            for sigma in [*np.arange(sigma_start, sigma_end, step), None]
        ]

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
