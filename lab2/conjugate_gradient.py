import numpy as np

from lab2.derivative import derivative
from lab2.one_dimension import fibonacci
from lab2.utils import default_stop_criterion


def conjugate_gradient(f, x0, optimizer=lambda task: fibonacci(task, 0, 1.1)[0], stop_criterion=None, grad=None):
    if grad is None:
        grad = derivative(f)
    if stop_criterion is None:
        stop_criterion = default_stop_criterion
    x = x0

    trace = [x]
    iterations = 0
    r = -grad(x)
    p = r

    while not stop_criterion(trace, grad, f):
        iterations += 1

        new_r = -grad(x)

        beta = new_r.dot(new_r) / r.dot(r)
        p = p * beta + new_r
        step = optimizer(lambda alpha: f(*(x + alpha * p)))
        x = x + step * p
        trace.append(x)

    return np.array(trace)
