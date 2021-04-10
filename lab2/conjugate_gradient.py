import numpy as np

from lab2.derivative import derivative
from lab2.one_dimension import fibonacci


def conjugate_gradient(f, x0, optimizer=lambda task: fibonacci(task, 0, 1)[0], stop_criterion=None, grad=None):
    if grad is None:
        grad = derivative(f)
    if stop_criterion is None:
        def stop_criterion(iterations, x):
            return iterations > 1000 or np.linalg.norm(x) > 1e9 or np.linalg.norm(grad(x)) < 1e-10
    x = x0

    trace = [x]
    iterations = 0
    r = -grad(x)
    p = r

    while not stop_criterion(iterations, x):
        iterations += 1

        new_r = -grad(x)

        beta = new_r.dot(new_r) / r.dot(r)
        p = p * beta + new_r
        step = optimizer(lambda alpha: f(*(x + alpha * p)))
        x = x + step * p
        trace.append(x)

    return np.array(trace)
