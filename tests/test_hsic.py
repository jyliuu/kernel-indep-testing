import unittest
from typing import Callable, Tuple

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from src import (
    simulate_dat2,
    sim_rho_going_to_0,
    simulate_p_values_resampling_from_HSIC,
    plot_xy_log,
    get_rho_steps,
    test_using_HSIC,
)
from src.dcor import test_using_dCor

P = 999
N = 500


def permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    test_method: Callable[[np.ndarray, np.ndarray], float],
    P: int,
) -> Tuple[np.ndarray, float, float]:
    N = len(X)
    T = test_method(X, Y)
    permutation_res = Parallel(n_jobs=-1)(
        delayed(test_method)(X, Y[np.random.permutation(N), :]) for _ in range(P)
    )

    extremes = (permutation_res > T).sum()

    return permutation_res, T, (1 + extremes) / (P + 1)


def plot_permutation_test(permutation_res, T, p_val, ax=None, T_offset=0.1) -> None:
    if ax is None:
        ax = plt.gca()

    ax.hist(permutation_res, bins=20)
    ax.axvline(x=T, color="red", linestyle="--", linewidth=2)
    ax.text(
        T + T_offset,
        max(ax.get_ylim()) * 0.9,
        f"p-value: {p_val:.4f}",
        fontsize=12,
        color="black",
    )


def compute_p_values(step):
    print("Computing for", step)
    p_vals = simulate_p_values_resampling_from_HSIC(
        lambda: simulate_dat2(N=N, p=10, q=10, rho=step), P=P
    )[0]
    return p_vals


methods = {
    "hsic_median": test_using_HSIC,
    "dcor": test_using_dCor,
}


def plot_permutation_test_for(method: str):
    rhos = np.arange(0, 0.09, 0.04)
    p = q = 3
    fig, axes = plt.subplots(1, len(rhos), figsize=(15, 4), sharex=True, sharey=True)

    for i, rho in enumerate(rhos):
        X, Y = simulate_dat2(N, rho=rho, p=p, q=q)
        permutation_res, T, p_val = permutation_test(X, Y, methods[method], P)
        ax = axes[i]
        plot_permutation_test(
            permutation_res, T, p_val, ax, (0.01, 0.1)[method == "dcor"]
        )
        ax.set_xlabel("Value", fontsize="large")
        ax.set_ylabel("Frequency", fontsize="large")
        ax.set_title(
            rf"Permutation distribution for $\rho={rho:.2f}$", fontsize="x-large"
        )

    plt.tight_layout()
    plt.savefig(f"../figures/permutation_dists/{method}_p{p}q{q}P{P}N{N}.pdf")
    plt.show()


class TestDCor(unittest.TestCase):
    def test_plot_permutation_hsic(self):
        plot_permutation_test_for("hsic_median")

    def test_plot_permutation_dcor(self):
        plot_permutation_test_for("dcor")

    def test_rho_qoing_to_0(self):
        steps, means, rejection_rates = sim_rho_going_to_0(
            sim_p_vals=compute_p_values, get_rho_steps=lambda: get_rho_steps(start=0.1)
        )
        plot_xy_log(steps, means, y_lab="p-value")
        plot_xy_log(steps, rejection_rates)


if __name__ == "__main__":
    unittest.main()
