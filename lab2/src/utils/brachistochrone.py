'''
Brachistochrone routines

`BrachistochroneApproximator` class wraps model creation and 
'''
import logging
import os
from scipy.integrate import quad
from functools import lru_cache
import numpy as np

from .methods.simpson import composite_simpson_ranged
from .methods.trapezoid import composite_trapezoid_ranged
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

def load_boundary_conds(filepath: str):
    log.debug(msg=f'Tries to load config from "{filepath}"')
    from json import loads
    with open(filepath, 'r') as confile:
        config = loads(confile.read())
    log.debug(msg=f'Config loaded')
    return config

class BrachistochroneApproximator:
    T_NODES = None
    INTEGRAND = None
    SETTINGS = None

    def __init__(self, filepath: str) -> None:
        self.SETTINGS = load_boundary_conds(filepath)

    def get_constants(self) -> tuple:
        log.debug(msg=f'Computes arbitrary constants of model (C and T)')
        ENDPOINT = (self.SETTINGS['X']['a'], self.SETTINGS['Y']['a'])
        log.debug(msg=f'The endpoint is {ENDPOINT}')

        C, T = find_constants(ENDPOINT)
        log.info(msg=f'Found constants: C = {C}, T = {T}')
        return C, T

    def set_parametrized_funcs(self, C: float):
        log.debug(msg=f'Sets functions for x and y for parameter t')

        funcs = (
            lambda t: C * (t - 0.5 * np.sin(2*t)),
            lambda t: C * (0.5 - 0.5 * np.cos(2*t))
        )
        return funcs

    def set_model(self):
        log.info(msg=f'Setting up model')

        C, T = self.get_constants()
        a = 1e-7
        b = T
        n = self.SETTINGS['N']['max']
        fx, fy = self.set_parametrized_funcs(C=C)

        X_NODES, Y_NODES, T_NODES = get_nodes_from_parameter(a=a, b=b, n=n, fy=fy, fx=fx)
        DY_NODES = get_derivatives_from_parameter(x_range=tuple(X_NODES), y_range=tuple(Y_NODES), t_range=tuple(T_NODES))
        
        def dx_generator():
            xdt = lambda t: C * (1 - np.cos(2*t))
            for i, _ in enumerate(T_NODES):
                yield xdt(T_NODES[i])

        Xdt_NODES = np.array([xdt for xdt in dx_generator()])

        INTEGRAND = get_integrand(y_range=tuple(Y_NODES), dy_range=tuple(DY_NODES), xdt_range=tuple(Xdt_NODES))

        self.T_NODES, self.INTEGRAND = T_NODES, INTEGRAND

        log.info(msg=f'Model is set up')
        log.info(msg=f'Collected parameters:\n\
            X nodes:\t{X_NODES} ({len(X_NODES)} items)\n\
            Y nodes:\t{Y_NODES} ({len(Y_NODES)} items)\n\
            dY nodes:\t{DY_NODES} ({len(DY_NODES)} items)\n\
            T nodes:\t{T_NODES} ({len(T_NODES)} items)\n\
            Integrand:\t{INTEGRAND} ({len(INTEGRAND)} items)')
    
    def compare_methods(self):
        log.info(msg=f'Compares trapezoid and Simpson methods and logs results')
        
        T_NODES, INTEGRAND = self.T_NODES, self.INTEGRAND
        C, T = self.get_constants()
        n = self.SETTINGS['N']['max']
        
        integral_values_trapezoid = []
        integral_values_simpson = []
        n_values = []
        divideby2g = lambda : np.sqrt(1 / 20)
        reference = np.sqrt(2 * C / 10) * T
        log.info(msg=f'Reference value is: {reference}')

        for nodes in range(3, 10001, 100):
            x_selected, y_selected = self.select_n(T_NODES, INTEGRAND, n=nodes)
            # log.debug(msg=f'Selected:\nt:{x_selected}\nIntegrand:{y_selected}')
            integral_value = composite_trapezoid_ranged(x_selected, y_selected, n=nodes)
            n_values.append(nodes)
            integral_values_trapezoid.append( reference - integral_value * divideby2g())
            log.info(msg=f'min F[y] computed for {nodes} nodes (trapezoid): {integral_value * divideby2g()}')
            integral_value = composite_simpson_ranged(x_selected, y_selected, n=nodes)
            integral_values_simpson.append( reference - integral_value * divideby2g())
            log.info(msg=f'min F[y] computed for {nodes} nodes (simpson): {integral_value * divideby2g()}')
        
        from .plotting import PlotArtist
        Artist = PlotArtist()
        Artist.add_log_plot(n_values, integral_values_trapezoid, style={
        'legend':'Trapezoid error',
        'color':'#67CC8E',
        'linestyle':'-'
        })
        Artist.add_log_plot(n_values, integral_values_simpson, style={
        'legend':'Simpson error',
        'color':'#9250BC',
        'linestyle':'-'
        })
        Artist.save_as('res/plots/integral')
        log.debug(msg=f'Saved integral figure')


        integral_all_nodes = composite_trapezoid_ranged(T_NODES, INTEGRAND, n=n)
        log.info(msg=f'min F[y] computed for all nodes (trapezoid): {integral_all_nodes * divideby2g()}')
        # log.info(msg=f'min F[y] computed for all nodes (trapezoid): {integral_all_nodes}')
        integral_all_nodes = composite_simpson_ranged(T_NODES, INTEGRAND, n=n)
        log.info(msg=f'min F[y] computed for all nodes (simpson): {integral_all_nodes * divideby2g()}')
        # log.info(msg=f'min F[y] computed for all nodes (simpson): {integral_all_nodes}')
        log.info(msg=f'Modeling finished. Plot is saved at "res/plots/integral.svg"')

        

    def select_n(self, x, y, n: int) -> tuple:
        if n < 1: raise ValueError
        h = (x[-1] - x[0]) / (n - 1)
        log.debug(msg=f'Hx = {h}')
        log.debug(msg=f'selecting {n} items')
        y_selected = []

        def n_items_generator():
            for i in range(n):
                # log.debug(msg=f'i = {i} looks for {x[0] + h * i}')
                nearest, i_nearest = find_nearest(x, x[0] + h * i)
                # log.debug(msg=f'Nearest is {nearest}')
                y_selected.append(y[i_nearest])
                yield nearest

        x_selected = np.array([x_i for x_i in n_items_generator()])
        y_selected = np.asarray(y_selected)
        return x_selected, y_selected


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
    return x_range, y_range, t_range

@lru_cache(maxsize=5)
def get_derivatives_from_parameter(x_range: tuple, y_range: tuple, t_range: tuple):
    log.debug(msg=f'Computes 1st derivative for nodes on given ranges')

    # dy_range = derivative1(x_nodes=t_range, y_nodes=y_range)
    # dx_range = derivative1(x_nodes=t_range, y_nodes=x_range)

    def derivative_generator():
        dydx = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
        for _, t in enumerate(t_range):
            yield dydx(t)

    derivative_range = np.array([d for d in derivative_generator()])
            
    log.debug(msg=f'Derivatives collected')
    return derivative_range
    # return dy_range, dx_range

@lru_cache(maxsize=5)
def get_integrand(y_range: tuple, dy_range: tuple, xdt_range: tuple) -> np.array:
    log.debug(msg=f'Computes integrand values for grid')

    def integrand_generator():
        integrand = lambda i: np.sqrt((1 + dy_range[i]**2) / y_range[i]) * xdt_range[i]
        for i, _ in enumerate(xdt_range):
            yield integrand(i)
    
    integrand_range = np.array([item for item in integrand_generator()])

    log.debug(msg=f'Integrand collected')
    return integrand_range
