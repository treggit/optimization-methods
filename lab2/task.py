import numpy as np
from matplotlib import pyplot as plt

from lab2.conjugate_gradient import conjugate_gradient
from lab2.newton import newton
from lab2.gradient_descent import grad_descent


def test_function1(x1, x2):
    return 100 * (x2 - x1) ** 2 + (1 - x1) ** 2


def rosenbrock(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def test_function2(x, y):
    return 2 * np.exp((-(x - 1) / 2) ** 2 - ((y - 1) / 2) ** 2) \
           + 3 * np.exp(-((x - 2) / 2) ** 2 - ((y - 3) / 2) ** 2)


initial_points = [
    np.array([1., 2.]),
    np.array([1., 0.]),
    np.array([-2., 2.]),
    np.array([2., 3.]),
    np.array([-0.1, 0.5])
]

for initial_point in initial_points:
    f = rosenbrock
    trace = conjugate_gradient(f, initial_point)

    print(len(trace), trace[-1])

# Task 3 (draft)

# derivative не умеет брать производную от np.exp, поэтому последней функции ниже пока нет, иначе падает.
# нужно это пофиксить (заменить np.exp?)

for [f_name, f] in [["f1", test_function1], ["rosenbrock", rosenbrock]]:
    for [solver_name, solver] in [
        ["conjugate gradient", conjugate_gradient],
        ["newton", newton],
        ["gradient descent", grad_descent]
    ]:
        points = solver(f, [0, 2])
        print(f_name, solver_name, points[-1])

# Task 4

initial_point = np.array([0, 2])
f = rosenbrock

xmin, xmax = -1.5, 1.5
ymin, ymax = -1, 3

x = np.arange(xmin, xmax, 0.1)
y = np.arange(ymin, ymax, 0.1)
x, y = np.meshgrid(x, y)
z = f(x, y)
plt.contour(x, y, z, 20)

for solver in [conjugate_gradient, newton]:
    points = solver(f, initial_point)
    plt.plot(points[:, 0], points[:, 1])
    plt.scatter(points[:, 0], points[:, 1])

plt.show()
