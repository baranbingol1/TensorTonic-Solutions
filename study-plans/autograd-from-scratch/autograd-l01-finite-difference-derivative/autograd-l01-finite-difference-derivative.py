import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    def f(y):
        l = len(coefficients) - 1
        sm = 0 
        while l >= 0:
            sm += coefficients[l]*y**l
            l -= 1
        return sm
    f_x = f(x)
    f_xplush = f(x+h)
    slope = (f_xplush - f_x) / h
    return (f_x, f_xplush, slope)