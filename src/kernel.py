import numpy as np
from sklearn.metrics import pairwise_distances


def gaussian_kernel_matrix(X, sigma=1):
    """
    Compute the Gaussian (RBF) kernel matrix given the data matrix X and sigma.

    Parameters:
        X (np.ndarray): The input data matrix (n_samples, n_features).
        sigma (float): The standard deviation (spread) of the Gaussian.

    Returns:
        np.ndarray: The Gaussian kernel (Gram) matrix.
    """
    # Compute the squared pairwise Euclidean distance matrix
    pairwise_sq_dists = pairwise_distances(X, metric='sqeuclidean')

    # Compute the Gaussian kernel matrix
    K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))

    return K

# other types of kernels?
