def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """

    def f_derivative(x): return 2*a*x + b
    x = x0
    for step in range(steps):
        x = x - lr * f_derivative(x)
    return x