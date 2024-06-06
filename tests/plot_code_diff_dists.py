import unittest
from functools import partial
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"] = True

from src import (
    permutation_test_p_val,
    test_using_HSIC,
    linear_kernel_matrix,
    get_p_vals_B_times,
    simulate_dat_m1,
    simulate_dat_m2,
    simulate_dat_m3,
    d_cov_test_p_val,
    gaussian_kernel_matrix,
)

N = 100
P = 299
B = 500

p_vals_methods = {
    "dCor": partial(d_cov_test_p_val, P=P),
    r"HSIC Gaussian ($\sigma=$median)": partial(
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


def simulate_for(
    sim_method,
    val_range: Iterable = range(1, 11),
    val_name="A",
    file_head="../data/m1_sim",
) -> np.ndarray:
    final_res = []
    for val in val_range:
        res_for_conf = []
        print("Testing", val)

        for name, method in p_vals_methods.items():
            print("Testing", name)
            p_vals = get_p_vals_B_times(
                method, partial(sim_method, N=N, **{val_name: val}), B=B, parallel=True
            )
            res_for_conf.append(p_vals)
        final_res.append(res_for_conf)

    final_res = np.array(final_res)
    np.save(f"{file_head}N{N}B{B}P{P}", final_res)

    return final_res


def plot_3_nice_plots(
    final_res: np.ndarray,
    param_name="A",
    param_mapping=lambda i: i,
    file_head="../figures/m1_dat/all_methods_m1",
) -> None:
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(-5, final_res.shape[0])  # Normalize A values to range 1 to 10
    handles = []  # To store legend handles
    labels = []  # To store legend labels

    for idx, (name, method) in enumerate(p_vals_methods.items()):
        ax = axes[idx]
        ax.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-")  # identity line
        for A in range(1, final_res.shape[0] + 1):
            p_values = final_res[A - 1][idx]
            x = np.sort(p_values)
            y = np.arange(len(x)) / float(len(x))
            color = cmap(norm(A))  # Get color from colormap
            (line,) = ax.plot(x, y, label=rf"${param_name}={A}$", color=color)

            if idx == 0:
                ax.set_ylabel(r"P($\hat{p} \leq \alpha)$")
                handles.append(line)
                labels.append(rf"${param_name}={param_mapping(A)}$")

        ax.set_title(name)
        ax.set_xlabel(r"$\alpha$")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(f"{file_head}N{N}B{B}P{P}.pdf", bbox_inches="tight")
    plt.show()


def plot_9_nice_plots(
    final_res_list: list[np.ndarray],
    param_names,
    param_mappings,
    file_head="../figures/9_nice_plots",
) -> None:
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(-5, final_res_list[0].shape[0])  # Normalize A values

    all_handles = []  # To store legend handles
    all_labels = []  # To store legend labels

    for row, final_res in enumerate(final_res_list):
        handles = []  # To store legend handles for this row
        labels = []  # To store legend labels for this row

        for idx, (name, method) in enumerate(p_vals_methods.items()):
            ax = axes[row, idx]
            ax.plot(
                np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-", linewidth=0.8
            )  # identity line
            for A in range(1, final_res.shape[0] + 1):
                p_values = final_res[A - 1][idx]
                x = np.sort(p_values)
                y = np.arange(len(x)) / float(len(x))
                color = cmap(norm(A))  # Get color from colormap
                (line,) = ax.plot(
                    x, y, label=rf"${param_names[row]}={A}$", color=color, linewidth=0.8
                )

                if idx == 0:
                    ax.set_ylabel(r"P($\hat{p} \leq \alpha)$")
                    handles.append(line)
                    labels.append(rf"${param_names[row]}={param_mappings[row](A)}$")
                if (
                    idx == len(p_vals_methods) - 1
                ):  # Check if it's the last subplot in the row
                    ax.legend(
                        handles,
                        labels,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        title=f"Dist. {row+1} - ${param_names[row]}$",
                    )

            if row == 0:
                ax.set_title(name)
            if row == len(final_res_list) - 1:
                ax.set_xlabel(r"$\alpha$")
            # Store handles and labels for the legend of this row
            if idx == 0:
                all_handles.append(handles)
                all_labels.append(labels)

    plt.tight_layout()
    plt.savefig(f"{file_head}N{N}B{B}P{P}.pdf", bbox_inches="tight")
    plt.show()


class TestCalibration2(unittest.TestCase):

    def test_plot_rejection_rates_for_m1_dist(self):
        final_res = simulate_for(simulate_dat_m1)
        # final_res = np.load(f"../data/m1_simN100B100P100.npy")
        plot_3_nice_plots(final_res)

    def test_plot_rejection_rates_for_m2_dist(self):
        final_res = simulate_for(
            simulate_dat_m2,
            val_name="rho",
            val_range=np.arange(0, 1.0, 0.1),
            file_head="../data/m2_sim",
        )
        # final_res = np.load(f"../data/m2_simN100B100P100.npy")

        plot_3_nice_plots(
            final_res,
            param_name=r"\rho",
            param_mapping=lambda i: (i - 1) / 10,
            file_head="../figures/m2_dat/all_methods_m2",
        )

    def test_plot_rejection_rates_for_m3_dist(self):
        final_res = simulate_for(
            simulate_dat_m3,
            val_name="a",
            file_head="../data/m3_sim",
        )
        plot_3_nice_plots(
            final_res,
            param_name=r"a",
            file_head="../figures/m3_dat/all_methods_m3",
        )

    def test_plot_everything_on_same_plot(self):
        m1_res = np.load(f"../data/m1_simN100B500P299.npy")
        m2_res = np.load(f"../data/m2_simN100B500P299.npy")
        m3_res = np.load(f"../data/m3_simN100B500P299.npy")

        plot_9_nice_plots(
            [m1_res, m2_res, m3_res],
            param_names=["A", r"\rho", "a"],
            param_mappings=[lambda i: i, lambda i: (i + 1) / 10, lambda i: i],
        )


if __name__ == "__main__":
    unittest.main()
