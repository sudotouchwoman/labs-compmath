import numpy as np

def composite_simpson(a: float, b: float, n: int, f) -> float:
    if a > b: raise ValueError
    if n < 3: raise ValueError
    if f is None: raise ValueError

    if n % 2 == 0: n -= 1

    h = (b - a) / (n - 1)
    x_range = np.linspace(a, b, n)

    def sum_terms():
        yield x_range[0]
        for i in range(1, n - 1):
            if i % 2 == 0:
                yield x_range[i]
                yield x_range[i]
            if i % 2 == 1:
                yield x_range[i]
                yield x_range[i]
                yield x_range[i]
                yield x_range[i]
        yield x_range[-1]

    approximation = (h / 3) * sum([f(term) for term in sum_terms()])
    return approximation
