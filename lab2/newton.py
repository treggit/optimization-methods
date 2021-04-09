import numpy as np

def newton(f, grad_f, hesse_f, x, optimizer, stop_criterion):
    iterations = 0
    trace = [x]

    while not stop_criterion(iterations, x):
        iterations += 1

        p = np.linalg.pinv(hesse_f(x)).dot(grad_f(x))
        x -= optimizer(lambda alpha: f(x - alpha * p)) * p
        trace.append(x)

    return trace