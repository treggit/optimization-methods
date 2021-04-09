def grad_descent(
        f, df, start,
        step_searcher=ternary_searcher,
        eps=1e-6,
        max_iters=100
):
    x = start
    prev = np.zeros_like(x)

    points = [np.append(start, f(start))]

    while abs(f(x) - f(prev)) > eps: #np.linalg.norm(x - prev) > eps:
        if max is not None and len(points) > max_iters:
            break

        dfx = df(x)
        dfx = dfx / np.sqrt(np.sum(dfx**2))
        step = step_searcher(lambda s: f(x - s * dfx), [-1, 1], len(points))
        prev = x
        x = x - step * dfx
        points.append(np.append(x, f(x)))

    return np.array(points)