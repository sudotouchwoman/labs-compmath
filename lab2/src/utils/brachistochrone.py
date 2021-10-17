import logging
import os
from numpy.core.fromnumeric import trace
from scipy.integrate import quad
from functools import lru_cache
import numpy as np

from .methods.simpson import composite_simpson
from .methods.trapezoid import composite_trapezoid
from .methods.diff import derivative1
from .optimizer import find_constants

LOGFILE = 'res/brachistochrone.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%D # %H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)
print(f'Writing log to {LOGFILE}')

def compare():
    f = lambda x: 5*x**2 - 3*x
    integral = composite_simpson(0, 1, 1000, f)
    log.info(msg=f'Simpson result: {integral}')
    integral = composite_trapezoid(0, 1, 1000, f)
    log.info(msg=f'Trapezoid result: {integral}')
    integral = quad(f, 0, 1)
    log.info(msg=f'Area is {integral[0]}')

def load_boundary_conds(filepath: str):
    log.debug(msg=f'Tries to load config from {filepath}')
    from json import loads
    with open(filepath, 'r') as confile:
        config = loads(confile.read())
    log.debug(msg=f'Config loaded')
    return config

def pretty_print_constants(src: dict):
    log.debug(msg=f'Tries to find constants (C and T)')
    endpoint = (src['X']['a'], src['Y']['a'])
    log.debug(msg=f'The endpoint is {endpoint}')
    C, T = find_constants(endpoint)
    log.info(msg=f'Found constants: C = {C}, T = {T}')

class BrachistochroneApproximator:
    X_NODES = None
    Y_NODES = None
    DY_NODES = None
    C = None
    T = None
    SETTINGS = None

    def __init__(self, filepath: str) -> None:
        self.SETTINGS = load_boundary_conds(filepath)

    def get_constants(self) -> tuple:
        log.debug(msg=f'Computes arbitrary constants of model (C and T)')
        endpoint = (self.SETTINGS['X']['a'], self.SETTINGS['Y']['a'])
        log.debug(msg=f'The endpoint is {endpoint}')

        C, T = find_constants(endpoint)
        log.info(msg=f'Found constants: C = {C}, T = {T}')
        return C, T

    def set_parametrized_funcs(self):
        log.debug(msg=f'Sets functions for x and y for parameter t')
        C = self.CT
        xa = self.SETTINGS['X']['a']
        ya = self.SETTINGS['Y']['a']

        funcs = (
            lambda t: (C * (t - 0.5 * np.sin(2*t)) - xa),
            lambda t: (C * (0.5 - 0.5 * np.cos(2*t)) - ya)
        )
        return funcs

    def set_model(self):
        log.info(msg=f'Setting up model')

        self.C, self.T = self.get_constants()
        a = 0
        b = self.T
        n = self.SETTINGS['N']['max']
        fx, fy = self.set_parametrized_funcs()
        
        self.X_NODES, self.Y_NODES = get_nodes_from_parameter(a=a, b=b, n=n, fy=fy, fx=fx)
        self.DY_NODES = get_derivatives_from_parameter(a=a, b=b, n=n, fy=fy, fx=fx)

        log.info(msg=f'Model is set up')

    def set_integrand(self):
        pass


def find_nearest(array: np.array, value: float) -> tuple:
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx

@lru_cache(maxsize=5)
def get_nodes_from_parameter(a: float, b: float, n: int, fy, fx) -> tuple:
    log.debug(msg=f'Fits nodes on given range [{a};{b}] ({n} nodes)')
    
    if a > b: raise ValueError
    if n < 2: raise ValueError
    if fy is None or fx is None: raise ValueError
    t_range = np.linspace(a, b, n)
    fy_v = np.vectorize(fy)
    fx_v = np.vectorize(fx)
    x_range = fx_v(t_range)
    y_range = fy_v(t_range)
    
    log.debug(msg=f'Node collections computed')
    return x_range, y_range

@lru_cache(maxsize=5)
def get_derivatives_from_parameter(a: float, b: float, n: int, fy, fx) -> np.array:
    log.debug(msg=f'Computes 1st derivative for nodes on given range [{a};{b}] ({n} nodes)')
    
    if a > b: raise ValueError
    if n < 2: raise ValueError
    if fy is None or fx is None: raise ValueError
    t_range = np.linspace(a, b, n)
    fy_v = np.vectorize(fy)
    fx_v = np.vectorize(fx)
    x_range = fx_v(t_range)
    y_range = fy_v(t_range)

    dy_range = derivative1(x_nodes=t_range, y_nodes=y_range)
    dx_range = derivative1(x_nodes=t_range, y_nodes=x_range)

    log.debug(msg=f'Derivatives collected')
    return dy_range / dx_range

@lru_cache(maxsize=5)
def get_integrand(y_range: np.ndarray, dy_range: np.ndarray, t_range: np.ndarray) -> np.ndarray:
    log.debug(msg=f'Computes integrand values for grid')

    def integrand_generator():
        integrand = lambda i: (1 + dy_range[i]**2) / y_range[i]
        for i, _ in enumerate(t_range):
            yield integrand(i)
    
    integrand_range = np.array([item for item in integrand_generator()])

    log.debug(msg=f'Integrand collected')
    return integrand_range