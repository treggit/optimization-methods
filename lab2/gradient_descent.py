import numpy as np

from lab2.one_dimension import ternary_searcher
from lab2.derivative import derivative


def grad_descent(
        f, x0,
        step_searcher=ternary_searcher,
        eps=1e-6,
        max_iters=100,
        df=None
):
    if df is None:
        df = derivative(f)
    x = x0
    prev = np.zeros_like(x)

    points = [x0]

    while abs(f(*x) - f(*prev)) > eps: #np.linalg.norm(x - prev) > eps:
        if max_iters is not None and len(points) > max_iters:
            break

        dfx = df(x)
        dfx = dfx / np.sqrt(np.sum(dfx**2))
        step = step_searcher(lambda s: f(*(x - s * dfx)), [-1, 1], len(points))
        prev = x
        x = x - step * dfx
        points.append(x)

    return np.array(points)
