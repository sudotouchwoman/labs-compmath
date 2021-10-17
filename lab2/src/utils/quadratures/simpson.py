import numpy as np

def composite_simpson(a: float, b: float, n: int, f) -> float:
    if a > b: raise ValueError
    if n < 1: raise ValueError
    if f is None: raise ValueError

    x_range = np.linspace(a, b, n)
    h = (b - a) / n

    def sum_terms():
        yield x_range[0]
        for i, item in enumerate(x_range[1:-1], 1):
            if i % 2 == 1:
                yield item
                yield item
            if i % 2 == 0:
                yield item
                yield item
                yield item
                yield item
        yield x_range[-1]

    approximation = (h / 3) * sum([f(term) for term in sum_terms()])
    return approximation
