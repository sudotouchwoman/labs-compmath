import numpy as np
from scipy.optimize import root

def solve_ode(x_0: list or np.ndarray, t_n: np.number, f, constraint=None, t_0=.0, h=5e-1, method='euler'):
    '''
    the function implements solvers for systems of ordinal differential equations, represented by a single
    vector-function of single independent variable

    + `x_0`: `list` or `np.ndarray` - the initial condition of dynamic system

    + `t_n`: `np.number` - the right bound independent variable value

    + `f`: function, the right part of the ODE

    + `constraint`: callable, is called to validate output of the method at each step

    + `t_0`: `np.number` - the left bound independent variable value

    + `h`: float, should be less then `t_n - t_0`, the computation step

    + `method`: one of (`euler`, `imp-euler`, `runge-kutta`). the method to use to obtain numeric solution of the ODE.
    Note that 'euler' has pretty bad accuracy (the worst, actually), 'imp-euler', representing backward (implicit) Euler method
    uses `root` to find numeric solution of the non-linear equation at each step thus is vulnerable to function shape
    'runge-kutta' utilizes 4-th order Runge-Kutta method which has greater accuracy but is significantly slower as it makes 4 calls
    to `f` per step
    '''

    methods = ('euler', 'imp-euler', 'runge-kutta')
    if method not in methods: raise ValueError(f'Invalid method name, expected one of {methods}')

    if t_n < t_0: raise ValueError(f'Invalid t bounds')
    if t_n < t_0 + h : raise ValueError(f'The step value is too big')
    if not callable(f): raise TypeError('f provided is not a callable')
    if constraint is not None and not callable(constraint): raise TypeError('User-provided constraint should be a callable')
    if constraint is None: constraint = lambda t, x: x

    x_0 = np.asarray(x_0)
    t_space = np.arange(t_0, t_n + h, step=h)
    f_space = np.zeros(shape=(len(t_space), len(x_0)))
    f_space[0] = x_0

    def euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            w = w + h*f(t, w)
            f_space[i+1] = constraint(t, w)
        return dict(t=t_space, y=f_space)

    def imp_euler():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]
            sol = root(lambda x: w - x + h*f(t, x), w, method='hybr')
            w =  sol.x[0] if len(sol.x) == 1 else sol.x
            f_space[i+1] = constraint(t, w)
        return dict(t=t_space, y=f_space)

    def runge_kutta():
        for i, t in enumerate(t_space[:-1]):
            w = f_space[i]

            k1 = h * f(t, w)
            k2 = h * f(t + .5*h, w + .5*k1)
            k3 = h * f(t + .5*h, w + .5*k2)
            k4 = h * f(t + h, w + k3)

            w = w + (k1 + 2*k2 + 2*k3 + k4) / 6
            f_space[i+1] = constraint(t, w)
        return dict(t=t_space, y=f_space)

    methods = dict(zip(methods, (euler, imp_euler, runge_kutta)))
    return methods[method]()


