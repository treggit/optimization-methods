import numpy as np

def default_stop_criterion(trace, grad, f):
    cur_point = trace[-1]
    iterations = len(trace)
    if iterations > 100:
        print("Iterations count is too large")
        return True
    if np.linalg.norm(cur_point) > 1e9:
        print("Points diverged")
        return True
    if np.linalg.norm(grad(cur_point)) < 1e-10:
        print("Gradient faded")
        return True

    if len(trace) > 1:
        delta = abs(f(*trace[-1]) - f(*trace[-2]))
        if delta < 1e-10:
            print("Update is too small: " + str(delta))
            return True

    return False