from math import sqrt


def ternary_searcher(f, lr, i):
    l, r = lr
    return ternary(f, l, r)[0]

def gold_searcher(f, lr, i):
    l, r = lr
    return golden(f, l, r)[0]

def fib_searcher(f, lr, i):
    l, r = lr
    return fibonacci_eps(f, l, r)[0]

def fixed_step(initial):
    return lambda f,lr,i: initial / i

def ternary(f, l, r, eps=2e-8):
    intervals = []
    while (r - l) / 2 >= eps:
        intervals += [[l, r]]
        m = (l + r) / 2
        l1 = m - eps / 2
        r1 = m + eps / 2

        if f(l1) > f(r1):
            l = l1
        else:
            r = r1

    return (l + r) / 2, intervals

def golden(f, a, b, eps=2e-8):
    fi = (1 + 5 ** 0.5) / 2
    intervals = []

    d = (b - a) / fi
    x1 = b - d
    x2 = a + d

    fx1, fx2 = f(x1), f(x2)

    while (b - a) / 2 >= eps:
        intervals += [[a, b]]
        if fx1 > fx2:
            a, b = x1, b
            x1 = x2
            x2 = a + (b - a) / fi

            fx1 = fx2
            fx2 = f(x2)
        else:
            a, b = a, x2
            x2 = x1
            x1 = b - (b - a) / fi

            fx2 = fx1
            fx1 = f(x1)

    return (a + b) / 2, intervals

def fibonacci(f, a, b, n=50):
    # n -= 1

    def fib(n):
        return 1 / sqrt(5) * (((1 + sqrt(5))/2) ** n - ((1 - sqrt(5))/2) ** n)

    f_n_2, f_n_1, f_n = fib(n - 2), fib(n - 1), fib(n)

    x1 = a + (b - a) * f_n_2 / f_n
    x2 = a + (b - a) * f_n_1 / f_n

    fx1 = f(x1)
    fx2 = f(x2)

    intervals = []
    for i in range(1, n - 2):
        intervals += [[a, b]]

        if fx1 > fx2:
            a = x1
            x1 = x2

            x2 = a + fib(n - i - 1) / fib(n - i) * (b - a)
            # x2 = b - (x1 - a)

            fx1 = fx2
            fx2 = f(x2)
        else:
            b = x2
            x2 = x1

            x1 = a + fib(n - i - 2) / fib(n - i) * (b - a)
            # x1 = a + (b - x2)

            fx2 = fx1
            fx1 = f(x1)

    return (x1 + x2) / 2, intervals


def fibonacci_eps(f, a, b, eps=2e-8):
    fib0, fib1 = 0, 1
    n = 1
    while (b - a) / fib1 > eps:
        fib0, fib1 = fib1, fib0 + fib1
        n += 1

    # n+=10

    # print(n, eps, (b - a) / fib1)
    return fibonacci(f, a, b, n)

