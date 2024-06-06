from functools import partial

import numpy as np

from src import (
    d_cov_test_p_val,
    permutation_test_p_val,
    test_using_HSIC,
    gaussian_kernel_matrix,
    linear_kernel_matrix,
    get_power_under,
)

P = 299
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


def main():
    final_res = get_power_under(p_vals_methods)
    np.save("./data/final_results.npy", final_res)


if __name__ == "__main__":
    main()
