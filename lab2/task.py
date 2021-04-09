import numpy as np

from lab2.conjugate_gradient import conjugate_gradient
from lab2.derivative import derivative
from lab2.one_dimension import fibonacci


def test_function1(x1, x2):
    return 100 * (x2 - x1) ** 2 + (1 - x1) ** 2


def rosenbrock(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def test_function2(x, y):
    return 2 * np.exp((-(x - 1) / 2) ** 2 - ((y - 1) / 2) ** 2) \
           + 3 * np.exp(-((x - 2) / 2) ** 2 - ((y - 3) / 2) ** 2)


def expanding(f):
    return lambda x: f(*x)


initial_points = [
    np.array([1., 2.]),
    np.array([1., 0.]),
    np.array([-2., 2.]),
    np.array([2., 3.]),
    np.array([-0.1, 0.5])
]

for initial_point in initial_points:
    f = expanding(rosenbrock)
    grad = derivative(rosenbrock)
    trace = conjugate_gradient(
        f,
        initial_point,
        optimizer=lambda task: fibonacci(task, 0, 1)[0],
        stop_criterion=lambda iterations, x:
            iterations > 1000
            or np.linalg.norm(x) > 1e9
            or np.linalg.norm(grad(x)) < 1e-10,
        grad=grad
    )

    print(len(trace), trace[-1])
