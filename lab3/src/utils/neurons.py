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

class Neuron:
    def __init__(self, is_excitatory:bool=False) -> None:
        self.is_excitatory = is_excitatory
        self.a = .02 if is_excitatory else .02 + .08 * rng.random()
        self.b = .2 if is_excitatory else .25 -.05 * rng.random()
        self.c = -65. + 15*rng.random()**2 if is_excitatory else -65.
        self.d = 8. - 6*rng.random()**2 if is_excitatory else 2.
        self.v = -65.
        self.u = -65.*self.b

    def I(self):
        return 5*rng.random() if self.is_excitatory else 2*rng.random()

    def W(self):
        return .5*rng.random() if self.is_excitatory else -rng.random()


class NeuralNetwork:
    def __init__(self, excitatory_ns: int = 800, total_ns: int = 1000) -> None:
        if excitatory_ns > total_ns: raise ValueError('Invalid number of neurons')
        if total_ns < 1: raise ValueError('Must be positive')

        inhibitory = np.repeat(False, total_ns - excitatory_ns)
        excitatory = np.repeat(True, excitatory_ns)

        log.info(msg=f'{total_ns - excitatory_ns} inhibitory neurons in total')
        log.info(msg=f'{excitatory_ns} excitatory neurons in total')

        isExcitatory = np.hstack((inhibitory, excitatory))
        self.neurons = [Neuron(is_excitatory=kind) for kind in isExcitatory]
        self.AdjMatrix = np.asarray( [[n.W() for n in self.neurons] for _ in range(total_ns)] )
        log.info(msg=f'Adjastency matrix:\n{self.AdjMatrix}')

    def simulate(self, t_n: float or int = 100., t_0: float or int = 0.,  h: float=5e-1, threshold: np.number=30) -> list:
        log.info(msg=f'Simulation started')
        if t_n < t_0: raise ValueError(f'Invalid t bounds')
        if t_n < t_0 + h : raise ValueError(f'The step value is too big')

        timings = np.arange(t_0, t_n + h, h)
        evals_per_timing = int( len(timings) / (t_n - t_0) )
        log.debug(msg=f'There will be {evals_per_timing} euler method iterations per milisecond')
        a = np.asarray([n.a for n in self.neurons])
        b = np.asarray([n.b for n in self.neurons])
        c = np.asarray([n.c for n in self.neurons])
        d = np.asarray([n.d for n in self.neurons])

        v = np.asarray([n.v for n in self.neurons])
        u = np.asarray([n.u for n in self.neurons])

        firings = []
        
        for timing in timings:
            I = np.asarray([n.I() for n in self.neurons])
            # log.debug(msg=f'Current (I) is {I}')

            fired = v > threshold
            fired_indices = np.where(v > threshold)[0]
            log.debug(msg=f'These neurons did not fire at ({timing}): {np.where( v < threshold)[0]}')
            # log.debug(msg=f'These neurons fired at ({timing}): {fired_indices}')
            for idx in fired_indices:
                firings.append({'time': timing, 'neuron': idx})
            v[fired] = c[fired]
            u[fired] += d[fired]
            log.debug(msg=f'V values: {v[::10]}')
            I += np.where(fired, self.AdjMatrix.sum(axis=1), 0)
            log.debug(msg=f'Current updated: {I[::10]}')

            log.debug(msg=f'Updates V and U')
            for _ in range(evals_per_timing):
                u += h*a*(b*v - u)
                v += h*(0.04*v**2 + 5*v - 140 + I)

        log.info(msg=f'Simulation finished')
        return firings

def draw_firings(firings: list, img_name: str='res/img/firings.svg') -> None:
    log.info(msg=f'Creates image')
    if len(firings) == 0: return
    # flatten = lambda l: [i for row in l for i in row]
    df_firings = pd.DataFrame(firings)
    fig = px.scatter(df_firings, x='time', y='neuron')
    fig.write_image(img_name)
    log.info(msg=f'SVG image saved at "{img_name}"')