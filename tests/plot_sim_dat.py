import functools
import unittest
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from src import simulate_dat_m1, simulate_dat_m2, simulate_dat_m3, simulate_dat2

plt.rcParams["text.usetex"] = True

N = 1000


def plot(X, Y, save_path=None):
    # Initialize a grid of subplots (p rows, q columns)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    # Create the grid plot
    for i in (0, 1):
        for j in (0, 1):
            ax = axes[i, j]  # Access the subplot
            ax.scatter(X[:, i], Y[:, j], alpha=0.5)
            ax.set_xlabel(f"X{i + 1}")
            ax.set_ylabel(f"Y{j + 1}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the title
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_heatmap(X, Y):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Create the grid plot
    for i in (0, 1):
        for j in (0, 1):
            ax = axes[i, j]  # Access the subplot
            heatmap, xedges, yedges = np.histogram2d(X[:, i], Y[:, j], bins=50)
            ax.imshow(
                heatmap.T,
                origin="lower",
                cmap="RdPu",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            )

            ax.set_xlabel(f"X{i + 1}")
            ax.set_ylabel(f"Y{j + 1}")

    fig.suptitle("Grid Plot of X vs Y")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the title
    plt.show()


def plot_heatmap_upper_left(X, Y, save_path=None):
    heatmap, xedges, yedges = np.histogram2d(X[:, 0], Y[:, 0], bins=40)

    # Plot heatmap
    plt.imshow(
        heatmap.T,
        origin="lower",
        cmap="magma",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    plt.colorbar(label="Counts")
    plt.xlabel("$X^2$")
    plt.ylabel("$Y^2$")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_upper_left_scatter(X, Y, save_path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], Y[:, 0], alpha=0.5)
    ax.set_xlabel("$X^2$")
    ax.set_ylabel("$Y^2$")

    if save_path:
        plt.savefig(save_path)
    plt.show()


class TestPlotSimDat(unittest.TestCase):

    def test_simulate_data2(self):
        p = q = 3
        rho = 0.2
        X, Y = simulate_dat2(N, p=p, q=q, rho=rho)

        # Initialize a grid of subplots (p rows, q columns)
        fig, axes = plt.subplots(p, q, figsize=(12, 8), sharex=True, sharey=True)

        # Create the grid plot
        for i in range(p):
            for j in range(q):
                ax = axes[i, j]  # Access the subplot
                cov_value = np.corrcoef(X[:, i], Y[:, j])[0, 1]
                ax.scatter(X[:, i], Y[:, j], alpha=0.5, s=5)
                ax.set_title(f"Corr.: {cov_value:.2f}", fontsize="x-large")
                ax.set_xlabel(f"X{i + 1}", fontsize="large")
                ax.set_ylabel(f"Y{j + 1}", fontsize="large")
                ax.tick_params(axis="both", which="major", labelsize="medium")

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for the title
        plt.savefig(f"../figures/simdat/gaussian/p{p}q{q}rho{rho}N{N}.pdf")
        plt.show()

    def test_simulate_m1(self):
        A = 9
        N = 1000
        X, Y = simulate_dat_m1(N, A=A)
        # plot(
        #     X,
        #     Y,
        #     save_path=f"../figures/simdat/dist2N{N}.pdf",
        # )

        # plot_heatmap_upper_left(X, Y, f"../figures/simdat/heat_map_m1N{N}.pdf")
        plot_upper_left_scatter(X, Y, f"../figures/simdat/scatter_m1N{N}A{A}.pdf")

    def test_simulate_m2(self):
        rho = 0.9
        N = 1000
        X, Y = simulate_dat_m2(N, rho=rho)
        # plot(
        #     X,
        #     Y,
        #     save_path=f"../figures/simdat/dist3N{N}.pdf",
        # )
        # plot_heatmap_upper_left(X, Y, f"../figures/simdat/heat_map_m2N{N}.pdf")
        plot_upper_left_scatter(X, Y, f"../figures/simdat/scatter_m2N{N}rho{rho}.pdf")

    def test_simulate_m3(self):
        a = 2
        N = 10000
        X, Y = simulate_dat_m3(N, a=a)

        plot_heatmap_upper_left(X, Y, f"../figures/simdat/heat_map_m3N{N}a{a}.pdf")


if __name__ == "__main__":
    unittest.main()
