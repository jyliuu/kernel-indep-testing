import numpy as np
from matplotlib import pyplot as plt


def plot_permutation_test(permutation_res, T, p_val) -> None:
    plt.hist(
        permutation_res, bins=40, edgecolor="black"
    )  # `bins` specifies the number of intervals
    plt.axvline(x=T, color="red", linestyle="--", linewidth=2)
    plt.text(
        T,
        max(plt.gca().get_ylim()) * 0.9,
        f"p-value: {p_val:.4f}",
        fontsize=12,
        color="black",
    )

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Data")
    plt.show()


def plot_p_values_under_line(p_vals, others, format_other=lambda x: x) -> None:
    plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), "k-")  # identity line

    x = np.sort(p_vals)
    y = np.arange(len(x)) / float(len(x))
    plt.plot(x, y, label=f"{format_other(others)}")
    plt.xlabel("Alpha")
    plt.ylabel("Fraction of Values ≤ Alpha")

    plt.legend()
    plt.show()


def plot_xy_log(
    rhos: object,
    pvals: object,
    log: object = True,
    x_lab: object = "Correlation: $rho$",
    y_lab: object = "Rejection rate",
    title: object = "Rejection rate vs $rho$",
) -> object:
    plt.figure(figsize=(10, 6))
    plt.plot(rhos, pvals, marker="o", linestyle="-", color="b")
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    if log:
        plt.xscale("log")
    plt.grid(True)
    plt.show()
