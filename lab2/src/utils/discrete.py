import numpy as np
import logging
import os

from .methods.linearinterp import LinearMethod
from .methods.simpson import composite_simpson
from .methods.trapezoid import composite_trapezoid
from .brachistochrone import NodeProvider
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

    def get_parametrized_funcs(self, C):
        log.debug(msg=f'Sets functions for x and y for parameter t')
        funcs = (
            lambda t: C * (t - 0.5 * np.sin(2*t)),
            lambda t: C * (0.5 - 0.5 * np.cos(2*t))
        )
        return funcs

    def create_interpolants(self, dtype = np.float64):

        C, T = self.SETTINGS['constants']['C'], self.SETTINGS['constants']['T']
        a = self.SETTINGS['lower-bound']
        b = T

        NodeSource = NodeProvider()
        NodeSource.set_constants(C, T)
        NodeSource.get_nodes_from_parameter()




    def create_error_surface(self):

        def interpolants_step():
            min = self.SETTINGS['integration-step']['min']
            max = self.SETTINGS['integration-step']['max']

            model = LinearMethod().fit()