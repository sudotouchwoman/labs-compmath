import numpy as np
import os
import logging
import pandas as pd
import plotly.express as px

# random numbers generator instance
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

def draw_firings(firings: list, img_name: str='res/img/firings', use_3D: bool=True) -> None:
    # use plotly express to create great vis with ease
    if len(firings) == 0: return
    log.info(msg=f'Creates image')

    df_firings = pd.DataFrame(firings)
    if use_3D:
        fig = px.scatter_3d(df_firings, x='Time', y='Neuron ID', z='Peak', color='State')
        fig.update_traces(marker=dict(size=2, opacity=.8))
        fig.write_html(f'{img_name}.html')
        log.info(msg=f'HTML scatter saved at "{img_name}.html"')
    else:
        fig = px.scatter(df_firings, x='Time', y='Neuron ID')
        fig.update_traces(marker=dict(size=2, opacity=.8))
        fig.write_image(f'{img_name}.svg', scale=5)
        log.info(msg=f'SVG image saved at "{img_name}.svg"')

def simulate(Ne: int=800, Ni: int=200, h: float=.5, t_0: float=0., t_n: float=1000., use_3D: bool=True) -> list:
    # simulate biological neural network activity
    # network consists of Ne excitatory neurons and Ni inhibitory neurons
    # there attributes are chosen with random term
    # the next state of the system is computed using forward Euler method
    # I found out (empirically) that the suitable simulation step lies within 0.2 and 0.5
    # if lower chosen, the system will encounter overflows (primarily in square, see below)
    # if higher chosen, the avalance of spiking occurs!

    if not t_0 < t_n: raise ValueError('bounds specified are invalid')
    if not .1 <= h <= 1.: raise ValueError('Invalid step size')
    evals = int( 1 / h )
    log.info(msg=f'Will perform {evals} evaluations per millisecond (step is {h})')

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

    AdjM = np.hstack((0.5*rng.random((Ne+Ni, Ne)), -rng.random((Ne+Ni, Ni))))
    firings = []
    
    log.debug(msg=f'Simulation starts')

    I_0 = np.hstack((5*rng.random(Ne), 2*rng.random(Ni)))
    thresh = 30

    def euler_step(v, u):
        _v = v + h*(0.04*v**2 + 5*v + 140 - u + I)
        _u = u + h*a*(b*v - u)
        return _v, _u

    for t in np.arange(t_0, t_n, step=h):
        fired = v > thresh

        if use_3D:
            for i, _ in filter(lambda e: e[1] and e[0] % 10 == 0, enumerate(fired)):
            #     # only append each 10-th firing neuron as there are like lots of these
            #     # to obtain all firing neurons, one should remove the second condition in filter
            #     # and better not collect other neuron data (it may look imbalanced on the same scatter plot)
                firings.append({'Time': t, 'Neuron ID': i, 'Peak': 30, 'State': 'Firing'})

            for i, f in filter(lambda e: e[1] <= thresh and e[0] % 10 == 0, enumerate(v)):
                # collect current data about all dormant neurons
                firings.append({'Time': t, 'Neuron ID': i, 'Peak': f, 'State': 'Dormant'})
        else:
            for i in np.where(v > thresh)[0]:
                # collect all firings
                firings.append({'Time': t, 'Neuron ID': i})

        v[fired] = c[fired]
        u[fired] += d[fired]

        I = I_0.copy()
        I += np.sum(AdjM[:, fired], axis=1)
        v, u = euler_step(v, u)

    log.debug(msg=f'Simulation finished')

    return firings
