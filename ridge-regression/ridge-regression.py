import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X, y = np.array(X), np.array(y).reshape(-1)
    mid_val = X.T @ X
    I = np.identity(mid_val.shape[0])
    return np.linalg.inv(mid_val + lam*I) @ X.T @ y