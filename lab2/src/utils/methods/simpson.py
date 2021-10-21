import numpy as np

def composite_simpson(a: float, b: float, n: int, f) -> float:
    if a > b: raise ValueError
    if n < 3: raise ValueError
    if f is None: raise ValueError

    x_range = np.linspace(a, b, n)
    h = (b - a) / (n - 1)

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

def composite_simpson_ranged(x_range: np.ndarray, y_range: np.ndarray, n: int) -> float:
    if n < 3: raise ValueError
    if y_range.shape != x_range.shape: raise ValueError

    if n % 2 == 0: n -= 1

    h = (x_range[-1] - x_range[0]) / (n - 1)

    def sum_terms_i():
        yield y_range[0]
        for i in range(1, n - 1):
            if i % 2 == 0:
                yield y_range[i]
                yield y_range[i]
            if i % 2 == 1:
                yield y_range[i]
                yield y_range[i]
                yield y_range[i]
                yield y_range[i]
        yield y_range[n - 1]

    approximation = (h / 3) * sum(list(sum_terms_i()))
    return approximation
    