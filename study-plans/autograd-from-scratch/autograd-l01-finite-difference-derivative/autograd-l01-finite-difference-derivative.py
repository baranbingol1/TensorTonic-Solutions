import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    c = np.asarray(coefficients, dtype=np.float64)
    x, h = np.float64(x), np.float64(h)

    def f(y): # horner's method
        acc = np.float64(0.0)
        for coef in reversed(c):
            acc = acc * y + coef
        return acc

    f_x = f(x)
    f_xh = f(x + h)
    slope = (f_xh - f_x) / h

    return (f_x, f_xh, slope)