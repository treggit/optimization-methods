from inspect import getfullargspec
from sympy import symbols, lambdify
import numpy as np

def derivative(f):
    args = symbols(getfullargspec(f).args)
    g = f(*args)
    return lambda x: np.array([lambdify(args, g.diff(arg))(*x) for arg in args])
