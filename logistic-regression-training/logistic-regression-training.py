import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape

    W = np.zeros(d, dtype=float)  # W shape: (d,)
    b = 0.0  # b is a scalar

    for _ in range(steps):
        z = X @ W + b  # z shape: (n,)
        p = _sigmoid(z)  # p shape: (n,)

        err = p - y  # err shape: (n,)

        grad_w = (X.T @ err) / n  # grad_w shape: (d,)
        grad_b = np.mean(err)  # grad_b is a scalar

        W = W - lr * grad_w  # W updated, shape remains (d,)
        b = b - lr * grad_b  # b updated, remains a scalar

    return W, b