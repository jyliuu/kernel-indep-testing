import numpy as np
from sklearn.metrics import pairwise_distances


def linear_kernel_matrix(X):
    X_normed = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]
    return X_normed @ X_normed.T, None


def gaussian_kernel_matrix(X, sigma=None):
    """
    Compute the Gaussian (RBF) kernel matrix given the data matrix X and sigma.

    Parameters:
        X (np.ndarray): The input data matrix (n_samples, n_features).
        sigma (float): The standard deviation (spread) of the Gaussian.

    Returns:
        np.ndarray: The Gaussian kernel (Gram) matrix.
    """
    # Compute the squared pairwise Euclidean distance matrix
    pairwise_sq_dists = pairwise_distances(X, metric="sqeuclidean")

    if not sigma:
        sigma = get_bandwidth(X)

    # Compute the Gaussian kernel matrix
    K = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    return K, sigma


def get_bandwidth(X):
    pairwise_sq_dists = pairwise_distances(X, metric="euclidean")
    optim_bandwidth = np.median(
        pairwise_sq_dists[np.tril_indices(pairwise_sq_dists.shape[0], k=-1)]
    )
    return optim_bandwidth
