import numpy as np


def simulate_dat(N=100, p=5, q=5, dependence=True):
    mean = np.zeros(p)
    A = np.random.randn(p, p)
    cov_matrix = A @ A.T
    X = np.random.multivariate_normal(mean, cov_matrix, N)

    if dependence:
        A = np.random.rand(p, q)  # generate some dependence
        Y = X @ A
    else:
        A = np.random.randn(q, q)
        cov_matrix = A @ A.T
        mean = np.zeros(q)
        Y = np.random.multivariate_normal(mean, cov_matrix, N)
    return X, Y


def simulate_dat2(N=100, p=2, q=2, rho=0.5, mean=0):
    X_cov = np.eye(p)
    Y_cov = np.eye(q)

    XY_cov = np.full((p, q), rho)
    top_row = np.hstack((X_cov, XY_cov))  # Horizontal stack: (A, C^T)
    bottom_row = np.hstack((XY_cov.T, Y_cov))  # Horizontal stack: (C, B)

    cov = np.vstack((top_row, bottom_row))

    sim = np.random.multivariate_normal(np.full(p + q, mean), cov, N)
    return sim[:, :p], sim[:, p:]


def simulate_dat_m1(N=100, A=5):
    XY_2 = np.random.uniform(0, 1, size=(N, 2))

    theta = np.random.uniform(0, 2 * np.pi, size=N)
    epsilon = np.random.normal(size=(N, 2))

    X1 = A * np.cos(theta) + epsilon[:, 1] / 4
    Y1 = A * np.sin(theta) + epsilon[:, 1] / 4

    return np.vstack((X1, XY_2[:, 0])).T, np.vstack((Y1, XY_2[:, 1])).T


def simulate_dat_m2(N=100, rho=0.5):
    XY_2 = np.random.uniform(0, 1, size=(N, 2))

    X1 = np.random.uniform(-1, 1, N)
    Y1 = np.abs(X1) ** rho * np.random.normal(size=N)

    return np.vstack((X1, XY_2[:, 0])).T, np.vstack((Y1, XY_2[:, 1])).T


def simulate_dat_m3(N=100, a=5):
    XY_2 = np.random.uniform(0, 1, size=(N, 2))
    X1 = np.random.uniform(-np.pi, np.pi, N)
    Y1 = []
    p_y_given_x = lambda y, x: 1 / (2 * np.pi) * (1 + np.sin(a * x) * np.sin(a * y))
    for x in X1:
        reject = True
        while reject:
            y = np.random.uniform(-np.pi, np.pi, size=1)
            U = np.random.uniform(0, 1, size=1)
            reject = p_y_given_x(y, x) < U
        Y1.append(y[0])
    return np.vstack((X1, XY_2[:, 0])).T, np.vstack((Y1, XY_2[:, 1])).T
