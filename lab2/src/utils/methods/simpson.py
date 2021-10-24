import numpy as np
import os
import logging

LOGFILE = 'res/discrete.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "False") == "True"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%D # %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

def composite_simpson(a: float, b: float, n: int, f) -> float:
    if a > b: raise ValueError
    if n < 3: raise ValueError
    if f is None: raise ValueError

    if n % 2 == 0: n += 1

    h = (b - a) / (n - 1)
    x_range = np.linspace(a, b, n)
    # log.info(msg=f'Computes simpson value for these: h = {h:e}')

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
    # log.info(msg=f'Approximation is {approximation:e}')
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
        yield y_range[-1]

    approximation = (h / 3) * sum(list(sum_terms_i()))
    return approximation
    