'''
This module provides interface for advanced part of the problem

Especially the one for discretization of the brachistochrone
by linear interpolation and composite simpson method

'''
import numpy as np
import logging
import os

from .methods.plotting import PlotArtist
from .methods.linearinterp import LinearMethod
from .methods.simpson import composite_simpson
from .error import BrachistochroneNodeProvider
from . import load_boundary_conds

LOGFILE = 'res/discrete.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%D # %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)
print(f'Writing log to {LOGFILE}')

class DiscreteOptimizer:
    def __init__(self, filepath: str) -> None:
        log.debug(msg=f'Tries to load config from "{filepath}"')
        self.SETTINGS = load_boundary_conds(filepath)
        log.debug(msg=f'Config loaded')

    def peek_plots(self):
        # draft to make sure everything worked out
        # use methods of `BrachistochroneNodeProvider` to produce nodes
        # use `LinearMethod` to interpolate
        log.info(msg=f'Quick test of performance: compute and plot single interpolant along with the actual curve')

        # these are merely sample test values, not a big deal to use others
        # the pipeline is pretty simular to one in `error` module
        # the only difference is that
        # now C and T are simply collected from config 
        # (it is assumed that their values were computed in previous steps)
        C, T = self.SETTINGS['C'], self.SETTINGS['T']
        a = self.SETTINGS['lower-bound']
        b = T
        n_int = 7
        n_ipl = 8
        nodes = 1e3

        # collect 1e3 nodes and select only n_ipl of them to fit the linear model
        # basically, I could just plot the selected nodes (this should yield the same result),
        # but here the very 'prediction' of the linear model was used
        fx, fy = BrachistochroneNodeProvider.get_parametrized_funcs(C=C)
        x_nodes, y_nodes, _ = BrachistochroneNodeProvider.get_nodes_from_parameter(a, b, nodes, fy=fy, fx=fx)
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(x_nodes, y_nodes, n_ipl)

        model = LinearMethod().fit(x_nodes=x_selected, y_nodes=y_selected)

        x_model = np.arange(x_nodes[0], x_nodes[-1], step=0.01)
        y_model = np.array([model.predict_y(x) for x in x_model])

        log.debug(msg=f'Computed, now plotting')
        
        artist = PlotArtist()
        artist.add_plot(x_nodes, y_nodes, style={
        'legend':'Brachistochrone',
        'color':'#962D3E',
        'linestyle':'-'
        })
        artist.add_plot(x_model, y_model, style={
        'legend':'Linear approximation',
        'color':'#343642',
        'linestyle':'--'
        })

        plot_name = 'res/plots/interpolant'
        artist.save_as(plot_name)
        
        log.info(msg=f'Plots saved at "{plot_name}.svg"')
        log.info(msg=f'Test finished')


class DiscreteModel:
    def __init__(self, x_nodes, y_nodes, n_int: int, n_ipl: int, G=9.8) -> None:
        self.X_NODES = np.asarray(x_nodes)
        self.Y_NODES = np.asarray(y_nodes)
        self.N_int = n_int
        self.N_ipl = n_ipl
        self.G = G

    def approximate(self, method=LinearMethod) -> float:
        X_NODES, Y_NODES = self.X_NODES, self.Y_NODES
        G = self.G
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(X_NODES, Y_NODES, self.N_ipl)
        model = method().fit(x_selected, y_selected)

        integrand = lambda x: np.sqrt( (1 + (model.predict_ydx(x)**2)) / model.predict_y(x) / 2 / G)
        
        approximation = composite_simpson(X_NODES[0], X_NODES[-1], self.N_int, integrand)
        return approximation
        