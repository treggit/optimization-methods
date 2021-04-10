import numpy as np

from lab2.one_dimension import ternary_searcher, fibonacci
from lab2.derivative import derivative
from lab2.utils import default_stop_criterion


def grad_descent(
        f, x0,
        optimizer=lambda task: fibonacci(task, 0, 1.1)[0],
        stop_criterion=None,
        df=None
):
    if df is None:
        df = derivative(f)
    if stop_criterion is None:
        stop_criterion = default_stop_criterion
    x = x0
    prev = np.zeros_like(x)

    points = [x0]

    while not stop_criterion(points, df, f): #np.linalg.norm(x - prev) > eps:

        dfx = df(x)
        dfx = dfx / np.sqrt(np.sum(dfx**2))
        step = optimizer(lambda s: f(*(x - s * dfx)))
        prev = x
        x = x - step * dfx
        points.append(x)

    return np.array(points)
