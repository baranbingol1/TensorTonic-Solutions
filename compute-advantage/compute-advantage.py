import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
        
    V = np.asarray(V, dtype=np.float64).reshape(-1)
    values = V[states]
    return returns - values

