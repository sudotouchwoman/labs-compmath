import numpy as np

def composite_trapezoid(a: float, b: float, n: int, f) -> float:
    if a > b: raise ValueError
    if n < 2: raise ValueError
    if f is None: raise ValueError

    x_range = np.linspace(a, b, n)
    h = (b - a) / (n - 1)

    def sum_terms():
        yield x_range[0]
        for i, item in enumerate(x_range[1:-1], 1):
            yield item
            yield item
        yield x_range[-1]

    approximation = (h / 2) * sum([f(term) for term in sum_terms()])
    return approximation
