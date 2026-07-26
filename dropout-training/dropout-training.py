import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)

    if p == 0:
        mask = np.ones(x.shape, dtype=float)
        return (x.astype(float), mask)
    
    random_vals = rng.random(x.shape)
    keep = (random_vals < (1-p))
    mask = np.where(keep, 1/(1-p), 0)
    x_dropped = x * mask
    return (x_dropped, mask)