import timeit
import unittest

from dcor import distance_covariance_sqr, distance_correlation_sqr

from src import (
    simulate_dat2,
    plot_permutation_test,
    get_p_vals_B_times,
    plot_p_values_under_line,
    simulate_p_values_resampling_from_dCor,
    plot_xy_log,
    sim_rho_going_to_0,
    permutation_test_p_val,
)
from src.dcor import test_using_dCor, V_n2

P = 100
N = 500


def plot_permuation_test_for_rho(rho, N, P):
    X, Y = simulate_dat2(N, rho=rho)

    _, T, _, permutation_res = permutation_test_p_val(X, Y, test_using_dCor, P)

    plot_permutation_test(T, permutation_res)


def compute_p_values(step):
    print("Computing for", step)
    p_vals = simulate_p_values_resampling_from_dCor(
        lambda: simulate_dat2(N=N, rho=step), P=P
    )
    return p_vals


class TestDCor(unittest.TestCase):

    def test_permutation_test(self):
        for rho in [0, 0.1, 0.2, 0.3]:
            plot_permuation_test_for_rho(rho, N, P)

    def test_p_values_under_line(self):
        p_vals, other_vars = get_p_vals_B_times(
            test_method=lambda X, Y: (distance_correlation_sqr(X, Y), None),
            simulate_dat=lambda: simulate_dat2(N, rho=0),
            P=P,
        )

        plot_p_values_under_line(p_vals, other_vars, format_other=lambda x: "")

    def test_p_values_under_line_dcor_lib(self):
        p_vals = simulate_p_values_resampling_from_dCor(
            lambda: simulate_dat2(N=N, rho=0), P=P
        )

        plot_p_values_under_line(p_vals, None, format_other=lambda x: "")

    def test_rho_qoing_to_0(self):
        steps, means, rejection_rates = sim_rho_going_to_0(sim_p_vals=compute_p_values)
        plot_xy_log(steps, means, y_lab="p-value")
        plot_xy_log(steps, rejection_rates)

    def test_dcor_equivalent(self):
        X, Y = simulate_dat2(N, rho=0.1)

        dCor_lib_res = distance_correlation_sqr(X, Y)
        dCor_self_impl = test_using_dCor(X, Y)

        print(dCor_lib_res, dCor_self_impl)
        self.assertAlmostEqual(dCor_lib_res, dCor_self_impl)

    def test_dist_cov_sqr_equivalent(self):
        X, Y = simulate_dat2(N, rho=0.1)
        Vn_lib = distance_covariance_sqr(X, Y)
        Vn_self = V_n2(X, Y)

        print(Vn_lib, Vn_self)
        self.assertAlmostEqual(Vn_lib, Vn_self)

    def test_dcor_timing(self):
        X, Y = simulate_dat2(N, rho=0.1)

        # Timing the library implementation
        lib_duration = (
            timeit.timeit(lambda: distance_correlation_sqr(X, Y), number=100) / 100
        )

        # Timing the custom implementation
        self_impl_duration = (
            timeit.timeit(lambda: test_using_dCor(X, Y), number=100) / 100
        )

        # Running once to get the results for assertion
        dCor_lib_res = distance_correlation_sqr(X, Y)
        dCor_self_impl = test_using_dCor(X, Y)

        print(
            f"Library implementation: {dCor_lib_res} (Average Time: {lib_duration:.6f} seconds)"
        )
        print(
            f"Custom implementation: {dCor_self_impl} (Average Time: {self_impl_duration:.6f} seconds)"
        )

        self.assertAlmostEqual(dCor_lib_res, dCor_self_impl, places=5)


if __name__ == "__main__":
    unittest.main()
