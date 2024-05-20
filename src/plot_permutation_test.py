from matplotlib import pyplot as plt


def plot_permutation_test(T, permutation_res) -> None:
    p_val = 1 - (permutation_res < T).mean()
    plt.hist(permutation_res, bins=40, edgecolor='black')  # `bins` specifies the number of intervals
    plt.axvline(x=T, color='red', linestyle='--', linewidth=2)

    plt.text(T, max(plt.gca().get_ylim())*0.9, f'p-value: {p_val:.4f}', fontsize=12, color='black')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()
