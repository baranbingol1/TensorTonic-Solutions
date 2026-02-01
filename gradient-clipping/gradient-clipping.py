import numpy as np

def clip_gradients(g, max_norm):
    if g is None:
        return None

    g = np.asarray(g)  # ensure ndarray output no matter what came in
    max_norm = float(max_norm)

    if max_norm <= 0:
        return g.copy()

    g64 = g.astype(np.float64, copy=False)
    norm = np.linalg.norm(g64)

    if norm <= max_norm or norm == 0.0:
        return g.copy()

    scale = max_norm / norm
    return (g64 * scale)