from lab2.derivative import derivative


def conjugate_gradient(f, x0, optimizer, stop_criterion, grad=None):
    if grad is None:
        grad = derivative(f)
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
        step = optimizer(lambda alpha: f(x + alpha * p))
        x += step * p
        trace.append(x)

    return trace
