import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    a, b, c, h = np.float64(a), np.float64(b), np.float64(c), np.float64(h)
    d = a*b + c
    d_a = (a+h)*b + c
    d_b = a*(b+h) + c
    d_c = a*b + (c+h)
    partial_a = (d_a - d) / h 
    partial_b = (d_b - d) / h 
    partial_c = (d_c - d) / h 

    return (float(d), float(partial_a), float(partial_b), float(partial_c))