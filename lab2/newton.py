import numpy as np

from lab2.derivative import derivative, hessian
from lab2.one_dimension import fibonacci
from lab2.utils import default_stop_criterion


def newton(f, x, optimizer=lambda task: fibonacci(task, 0, 1.1)[0], stop_criterion=None, grad=None, hesse=None):
    if grad is None:
        grad = derivative(f)
    if hesse is None:
        hesse = hessian(f)
    if stop_criterion is None:
        stop_criterion = default_stop_criterion
    iterations = 0
    trace = [x]

    while not stop_criterion(trace, grad, f):
        iterations += 1

        p = np.linalg.pinv(hesse(x)).dot(grad(x))
        x = x - optimizer(lambda alpha: f(*(x - alpha * p))) * p
        trace.append(x)

    return np.array(trace)
