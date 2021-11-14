import numpy as np
from scipy.optimize import root
import os
import logging
import pandas as pd
import plotly.express as px

rng = np.random.default_rng()

LOGFILE = 'res/logs/neurons.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%D # %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)

def solve_neuron_ode(x_0: list or np.ndarray, t_n: np.number, f, isspike, reset, t_0=.0, h=5e-1, method='euler'):
    # solve system of ode-s describing the biological neuron
    # 3 methods are implemented: vanilla forward Euler, backward Euler with
    # numeric solution of non-linear system
    # and Runge-Kutta method (4th order)
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
            # there was an idea of predictor-corrector scheme
            # but the results were unsatisfactory
            # predicted = w + h*f(t, w)
            # corrected = w + h*f(t, predicted)
            # f_space[i+1] = corrected
            nonlin = lambda W: w + h*f(t, W) - W
            sol = root(nonlin, w)
            f_space[i+1] = sol.x
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


def draw_firings(firings: list, img_name: str='res/img/firings.svg') -> None:
    # use plotly express to create great vis with ease
    log.info(msg=f'Creates image')
    if len(firings) == 0: return
    df_firings = pd.DataFrame(firings)
    fig = px.scatter_3d(df_firings, x='time', y='neuron ID', z='peak', color='color')
    fig.update_traces(marker=dict(size=3, opacity=.7))
    fig.write_image(img_name)
    fig.write_html("res/img/surface.html")
    log.info(msg=f'SVG image saved at "{img_name}"')

def simulate(Ne: int=800, Ni: int=200) -> list:
    # simulate biological neural network activity
    # network consists of Ne excitatory neurons and Ni inhibitory neurons
    # there attributes are chosen with random term
    # the next state of the system is computed using forward Euler method
    # I found out (empirically) that the suitable simulation step lies within 0.2 and 0.5
    # if lower chosen, the system will encounter overflows (primarily in square, see below)
    # if higher chosen, the avalance of spiking occurs!

    # uncomment to use same random variables for all neuron attributes
    # re = rng.random(Ne)
    # ri = rng.random(Ni)
    # a = np.hstack((0.02*np.ones(Ne), 0.02 + 0.08*ri))
    # b = np.hstack((0.2*np.ones(Ne), 0.25 - 0.05*ri))
    # c = np.hstack((-65.0 + 15.0*re**2, -65.0*np.ones(Ni)))
    # d = np.hstack((8.0 - 6*re**2, 2*np.ones(Ni)))

    log.info(msg=f'Network: {Ne} excitatory neurons and {Ni} inhibitory')
    a = np.hstack((0.02 * np.ones(Ne), 0.02 + 0.08*rng.random(Ni)))
    b = np.hstack((0.2 * np.ones(Ne), 0.25 - 0.05*rng.random(Ni)))
    c = np.hstack((-65.0 + 15.0*rng.random(Ne)**2, -65.0*np.ones(Ni)))
    d = np.hstack((8 - 6*rng.random(Ne)**2, 2.0*np.ones(Ni)))

    v = -65.0 * np.ones_like(a)
    u = v * b

    h = .5
    evals = int( 1 / h )
    log.info(msg=f'Will perform {evals} evaluations per millisecond (step is {h})')

    AdjM = np.hstack((0.5*rng.random((Ne+Ni, Ne)), -rng.random((Ne+Ni, Ni))))
    firings = []
    color = lambda p: 'red' if p > 30 else 'blue'

    for t in range(1000):
        fired = v > 30
        if t % 10 == 0:
            log.debug(msg=f'V values fired: {v[fired][::2]}')
        # for f in np.where(v > 30)[0]:
        #     firings.append({'time': t, 'neuron': f, 'peak': min(v[f], 1e2)})
        for i, f in enumerate(v[::10]):
            firings.append({'time': t, 'neuron ID': i, 'peak': min(f, 2e2), 'color': color(f)})
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]

        I = np.hstack((5*rng.random(Ne), 2*rng.random(Ni)))
        I += np.sum(AdjM[:, fired], axis=1)

        for _ in range(evals):
            u += h*a*(b*v - u)
            v += h*(0.04*v**2 + 5*v + 140 - u + I)

    return firings
