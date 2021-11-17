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

def draw_firings(firings: list, img_name: str='res/img/firings.svg') -> None:
    # use plotly express to create great vis with ease
    log.info(msg=f'Creates image')
    if len(firings) == 0: return
    df_firings = pd.DataFrame(firings)
    fig = px.scatter_3d(df_firings, x='Time', y='Neuron ID', z='Peak', color='State')
    # fig = px.scatter(df_firings, x='time', y='neuron ID')
    fig.update_traces(marker=dict(size=2, opacity=.8))
    # fig.write_image(img_name, scale=5)
    fig.write_html("res/img/surface-fixed.html")
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

    for t in range(1000):
        fired = v > 30
        # for f in np.where(v > 30)[0]:
        #     firings.append({'time': t, 'neuron ID': f})
        v[fired] = c[fired]
        u[fired] += d[fired]

        for i, spike in enumerate(fired):
            if spike and i % 10 == 0:
                firings.append({'Time': t, 'Neuron ID': i, 'Peak': 30, 'State': 'Firing'})

        for i, f in enumerate(v[::10]):
            if f > 30: continue
            firings.append({'Time': t, 'Neuron ID': i*10, 'Peak': f, 'State': 'Dormant'})

        I = np.hstack((5*rng.random(Ne), 2*rng.random(Ni)))
        I += np.sum(AdjM[:, fired], axis=1)

        for _ in range(evals):
            _v = v + h*(0.04*v**2 + 5*v + 140 - u + I)
            _u = u + h*a*(b*v - u)
            v = _v
            u = _u

    return firings
