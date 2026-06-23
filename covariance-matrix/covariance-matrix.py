import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    
    x_shape = np.shape(X)
    if x_shape[0] <= 1 or len(x_shape) != 2:
        return None
        
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    
    return (1 / (np.shape(X)[0] - 1)) * (X_centered.T @ X_centered)