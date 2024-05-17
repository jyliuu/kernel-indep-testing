import unittest

import numpy as np
from matplotlib import pyplot as plt

from src import calibration, get_p_values_vs_uniform, gaussian_kernel_matrix


class TestCalibration(unittest.TestCase):
    def test_calibration(self):
        res = calibration()
        plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'k-')  # identity line

        for sigma, p_values in res:
            x = np.sort(p_values)
            y = np.arange(len(x)) / float(len(x))
            plt.plot(x, y, label=f'sigma = {sigma:.2f}')
        plt.xlabel('Alpha')
        plt.ylabel('Fraction of Values ≤ Alpha')
        plt.legend()  # Display the legend with the sigma label
        plt.show()

    def test_gaussian_using_median_bandwidth(self):
        p_values, sigmas = get_p_values_vs_uniform(lambda X: gaussian_kernel_matrix(X, sigma=0.51), B=100)

        p_values_mean = np.mean(p_values)
        sigmas_avg = np.mean(sigmas, axis=0)

        plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'k-')  # identity line
        x = np.sort(p_values)
        y = np.arange(len(x)) / float(len(x))
        plt.plot(x, y, label=f'sigma = ({sigmas_avg[0]:.2f}, {sigmas_avg[1]:.2f}), p_value_mean = {p_values_mean}')
        plt.xlabel('Alpha')
        plt.ylabel('Fraction of Values ≤ Alpha')
        plt.legend()  # Display the legend with the sigma label
        plt.show()

        print(np.mean(p_values), np.mean(sigmas))

if __name__ == '__main__':
    unittest.main()
