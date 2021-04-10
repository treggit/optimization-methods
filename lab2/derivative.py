from inspect import getfullargspec
from sympy import symbols, lambdify
import numpy as np


def derivative(f):
    args = symbols(getfullargspec(f).args)
    g = f(*args)
    return lambda x: np.array([lambdify(args, g.diff(arg))(*x) for arg in args])


def hessian(f):
    args = symbols(getfullargspec(f).args)
    g = f(*args)
    return lambda x: np.array([[lambdify(args, g.diff(arg1).diff(arg2))(*x) for arg1 in args] for arg2 in args])
