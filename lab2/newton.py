import numpy as np

from lab2.derivative import derivative, hessian
from lab2.one_dimension import fibonacci


def newton(f, x, optimizer=lambda task: fibonacci(task, 0, 1)[0], stop_criterion=None, grad=None, hesse=None):
    if grad is None:
        grad = derivative(f)
    if hesse is None:
        hesse = hessian(f)
    if stop_criterion is None:
        def stop_criterion(iterations, x):
            return iterations > 1000 or np.linalg.norm(x) > 1e9 or np.linalg.norm(grad(x)) < 1e-10
    iterations = 0
    trace = [x]

    while not stop_criterion(iterations, x):
        iterations += 1

        p = np.linalg.pinv(hesse(x)).dot(grad(x))
        x = x - optimizer(lambda alpha: f(*(x - alpha * p))) * p
        trace.append(x)

    return np.array(trace)
