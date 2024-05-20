import unittest

from src import simulate_dat2, permutation_test, plot_permutation_test, test_using_HSIC_


def plot_permuation_test_for_rho(rho, N, P):
    X, Y = simulate_dat2(N, rho=rho)

    _, T, _, permutation_res = permutation_test(
        X, Y,
        test_using_HSIC_,
        P
    )

    plot_permutation_test(T, permutation_res)


class TestDCor(unittest.TestCase):
    P = 5000
    N = 100

    def test_permutation_test(self):
        for rho in [0, 0.1, 0.2, 0.3]:
            plot_permuation_test_for_rho(rho, TestDCor.N, TestDCor.P)

if __name__ == '__main__':
    unittest.main()
