import unittest
import matplotlib.pyplot as plt
import numpy as np

from src import simulate_dat2


class TestSimulateData(unittest.TestCase):
    def test_simulate_data2(self):
        p = q = 5
        X, Y = simulate_dat2(p = p, q = q, rho=0.1)

        # Initialize a grid of subplots (p rows, q columns)
        fig, axes = plt.subplots(p, q, figsize=(12, 8))

        # Create the grid plot
        for i in range(p):
            for j in range(q):
                ax = axes[i, j]  # Access the subplot
                if i == j:
                    # On the diagonal, show covariance
                    cov_value = np.cov(X[:, i], Y[:, j])[0, 1]
                    ax.text(0.5, 0.5, f"Cov: {cov_value:.2f}", fontsize=12, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Off-diagonal, create a scatter plot
                    ax.scatter(X[:, i], Y[:, j], alpha=0.5)
                ax.set_xlabel(f'X{i + 1}')
                ax.set_ylabel(f'Y{j + 1}')

        fig.suptitle("Grid Plot of X vs Y with Covariance on Diagonal")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the title
        plt.show()



if __name__ == '__main__':
    unittest.main()
