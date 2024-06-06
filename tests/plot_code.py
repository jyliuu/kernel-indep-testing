import functools
import unittest
from functools import partial

import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.unicode_minus": True,
    }
)
width = 6

from src import (
    calibration,
    get_p_values_vs_uniform,
    gaussian_kernel_matrix,
    get_rho_steps,
    simulate_dat2,
    test_using_HSIC,
    linear_kernel_matrix,
    permutation_test_p_val,
    d_cov_test_p_val,
    get_p_vals_B_times,
    get_bandwidth,
)


p = q = 2
N = 500
P = 299
B = 100


def get_rejection_rates_for_methods_at_rho(rho, p_vals_methods, B, N, p, q):
    def to_run():
        print("Testing for", rho)
        X, Y = simulate_dat2(N, p, q, rho)

        return [method(X, Y) for method in p_vals_methods]

    res_rho = Parallel(n_jobs=-1)(delayed(to_run)() for _ in range(B))
    return np.mean(np.array(res_rho) < 0.05, axis=0)


def get_power_under(
    p_vals_methods,
    dims=[2, 10, 100, 150],
    rhos=[0.1, 0.05, 0.01, 0.006],
    add_sigma_scaled=False,
):
    final_res = []
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, (d, rho) in enumerate(zip(dims, rhos)):
        ax = axs[i]
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
                    P=P,
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

        ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-")  # identity line

        for name, p_values in zip(methods.keys(), res_for_conf):
            if name == "HSIC Gaussian (median bandwidth)":
                print("Getting median bandwidth")
                X, Y = simulate_dat2(N=10**4, p=d, q=d, rho=rho)
                X_bw = get_bandwidth(X)
                Y_bw = get_bandwidth(Y)
                print(f"Median for d={d}, rho={rho}, {X_bw}, {Y_bw}")
                name = rf"Median bandwidth: $\sigma \approx {(X_bw+Y_bw)/2:.2f}$"
            x = np.sort(p_values)
            y = np.arange(len(x)) / float(len(x))
            ax.plot(x, y, label=rf"{name}")

        ax.set_xlabel(r"$\alpha$", fontsize="large")
        ax.set_ylabel(r"P($\hat{p} \leq \alpha)$", fontsize="large")
        ax.set_title(
            (rf"Power under $d={d}, \rho={rho}$", rf"Null rejection rate $d={d}$")[
                not rho
            ],
            fontsize="large",
        )
        ax.legend(loc="lower right")  # Display the legend with the sigma label

    plt.tight_layout()
    plt.savefig(
        f"../figures/increasing_dim/{'independence/' if not sum(rhos) else ''}combined_plots_diff_methods_N{N}P{P}B{B}.pdf"
    )
    plt.show()

    final_res = np.array(final_res)
    return final_res


p_vals_methods = {
    "dCor": partial(d_cov_test_p_val, P=P),
    r"HSIC Gaussian (median bandwidth)": partial(
        permutation_test_p_val,
        test_method=partial(test_using_HSIC, kernel=gaussian_kernel_matrix),
        P=P,
    ),
    "HSIC linear": partial(
        permutation_test_p_val,
        test_method=partial(test_using_HSIC, kernel=linear_kernel_matrix),
        P=P,
    ),
}


class TestCalibration(unittest.TestCase):
    def test_show_bandwidths_matters_d_equals_10(self):
        X, Y = simulate_dat2(N * 100, p, q, rho=0)
        print("Median bandwidths", get_bandwidth(X), get_bandwidth(Y))

        res = calibration(
            p=p,
            q=q,
            N=N,
            P=P,
            B=B,
            sigma_start=1,
            sigma_end=8,
            step=1,
            parallel=True,
        )
        plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-")  # identity line

        for sigma, p_values in res:
            x = np.sort(p_values)
            y = np.arange(len(x)) / float(len(x))
            plt.plot(
                x,
                y,
                label=rf"$\sigma=" + (f"{sigma or 0:.2f}", "median")[not sigma] + "$",
                linewidth=1,
            )
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"P($\hat{p} \leq \alpha)$")
        plt.legend()  # Display the legend with the sigma label
        plt.savefig(f"../figures/calibration/calibration_p{p}q{q}N{N}B{B}.pdf")
        plt.show()

    def test_gaussian_using_median_bandwidth(self):
        p_values = get_p_values_vs_uniform(
            lambda X: gaussian_kernel_matrix(X, sigma=0.51), B=100
        )

        p_values_mean = np.mean(p_values)

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-")

        x = np.sort(p_values)
        y = np.arange(len(x)) / float(len(x))
        ax.plot(x, y, label=f"p_value_mean = {p_values_mean:.2f}")

        ax.set_xlabel(rf"$\alpha$")
        ax.set_ylabel(r"P($\hat{p} \leq \alpha)$")

        ax.legend()
        plt.savefig("../figures/test.pdf", dpi=6600)
        plt.show()

    def test_plot_multiple_methods_rho_going_to_0(self):
        rho_start = 0.5
        rho_end = 0.01
        steps = get_rho_steps(start=rho_start, end=rho_end, n_steps=10)
        to_call = functools.partial(
            get_rejection_rates_for_methods_at_rho,
            p_vals_methods=p_vals_methods.values(),
            N=N,
            p=p,
            q=q,
            B=B,
        )

        results = []
        for step in steps:
            results.append(to_call(step))
        results = np.array(results)

        plt.figure(figsize=(8, 4))
        for i in range(results.shape[1]):
            plt.plot(
                steps,
                results[:, i],
                marker="o",
                linestyle="-",
                label=f"{list(p_vals_methods.keys())[i]}",
            )
        plt.xlabel(r"Correlation: $\rho$")
        plt.ylabel("Rejection rate")
        plt.xscale("log")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            f"../figures/diff_methods_vs_rho_N{N}B{B}p{p}q{q}start{rho_start}end{rho_end}.pdf"
        )
        plt.show()

    def test_plot_increasing_dimension(self):
        final_res = get_power_under(p_vals_methods)


if __name__ == "__main__":
    unittest.main()
