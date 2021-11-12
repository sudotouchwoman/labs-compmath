import numpy as np
from scipy.optimize import root
import os
import logging

LOGFILE = 'res/logs/odeint.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%D # %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

def solve_ode(x_0: list or np.ndarray, t_n: np.number, f, constraint=None, t_0=.0, h=5e-1, method='euler'):
    methods = ('euler', 'imp-euler', 'runge-kutta')
    if method not in methods: raise ValueError(f'Invalid method value, expected one of {methods}')

    if t_n < t_0: raise ValueError(f'Invalid t bounds')
    if t_n < t_0 + h : raise ValueError(f'The step value is too big')
    if not callable(f): raise TypeError('f provided is not a callable')

    x_0 = np.asarray(x_0)
    t_space = np.arange(t_0, t_n + h, step=h)
    f_space = np.zeros(shape=(len(t_space), len(x_0)))
    f_space[0] = x_0

    def euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if callable(constraint): w = constraint(t, w)
            f_space[i+1] = w + h*f(t, w)
        return dict(t=t_space, y=f_space)


    def imp_euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if callable(constraint): w = constraint(t, w)
            nonlin = lambda x: w + h*f(t, x) - x
            sol = root(nonlin, w)
            f_space[i+1] = sol.x[0] if len(sol.x) == 1 else sol.x
        return dict(t=t_space, y=f_space)


    def runge_kutta():
        
        k1 = lambda t, w: (h * f(t, w))
        k2 = lambda t, w: h * f(t + .5*h, (w + .5*k1(t, w)))
        k3 = lambda t, w: h * f(t + .5*h, (w + .5*k2(t, w)))
        k4 = lambda t, w: h * f(t + h, (w + k3(t, w)))

        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if callable(constraint): w = constraint(t, w)
            f_space[i+1] = w + (k1(t, w) + 2*k2(t, w) + 2*k3(t, w) + k4(t, w)) / 6
        return dict(t=t_space, y=f_space)

    methods = dict(zip(methods, (euler, imp_euler, runge_kutta)))
    return methods[method]()


def solve_neuron_ode(x_0: list or np.ndarray, t_n: np.number, f, isspike, reset, t_0=.0, h=5e-1, method='euler'):
    methods = ('euler', 'imp-euler', 'runge-kutta')
    if method not in methods: raise ValueError(f'Invalid method value, expected one of {methods}')

    if t_n < t_0: raise ValueError(f'Invalid t bounds')
    if t_n < t_0 + h : raise ValueError(f'The step value is too big')
    if not callable(f): raise TypeError('f provided is not a callable')

    x_0 = np.asarray(x_0)
    t_space = np.arange(t_0, t_n + h, step=h)
    f_space = np.zeros(shape=(len(t_space), len(x_0)))
    f_space[0] = x_0

    def euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if isspike(w):
                f_space[i+1] = reset(w)
                continue
            f_space[i+1] = w + h*f(t, w)
        return dict(t=t_space, y=f_space)

    def imp_euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if isspike(w):
                f_space[i+1] = reset(w)
                continue
            nonlin = lambda W: w + h*f(t, W) - W
            sol = root(nonlin, w)
            f_space[i+1] = sol.x[0] if len(sol.x) == 1 else sol.x
        return dict(t=t_space, y=f_space)

    def runge_kutta():

        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            if isspike(w):
                f_space[i+1] = reset(w)
                continue

            k1 = h*f(t,w)
            k2 = h*f(t + .5*h, w + .5*k1)
            k3 = h*f(t + .5*h, w + .5*k2)
            k4 = h*f(t + h, w + k3)

            f_space[i+1] = w + (k1 + 2*k2 + 2*k3 + k4) / 6
        return dict(t=t_space, y=f_space)

    methods = dict(zip(methods, (euler, imp_euler, runge_kutta)))
    return methods[method]()