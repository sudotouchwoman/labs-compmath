'''
Brachistochrone routines

`BrachistochroneApproximator` class wraps model creation and 
'''
import logging
import os
from functools import lru_cache
from re import T
import numpy as np

from .methods.simpson import composite_simpson_ranged
from .methods.trapezoid import composite_trapezoid_ranged
from .optimizer import find_constants, find_upper_bound

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
    SETTINGS = None

    def __init__(self, filepath: str) -> None:
        self.SETTINGS = load_boundary_conds(filepath)

    def set_model(self):
        log.info(msg=f'Setting up model')

        def get_constants() -> tuple:
            log.debug(msg=f'Computes arbitrary constants of model (C and T)')

            boundary_cond = self.SETTINGS['Boundary-condition']
            ENDPOINT = (boundary_cond['x'], boundary_cond['y'])
            log.debug(msg=f'The endpoint is {ENDPOINT}')

            C, T = find_constants(ENDPOINT)
            log.info(msg=f'Found constants: C = {C}, T = {T}')
            return C, T

        def get_parametrized_funcs():
            log.debug(msg=f'Sets functions for x and y for parameter t')
            funcs = (
                lambda t: C * (t - 0.5 * np.sin(2*t)),
                lambda t: C * (0.5 - 0.5 * np.cos(2*t))
            )
            return funcs

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
        def get_ydx_from_parameter(t_range: tuple):
            log.debug(msg=f'Computes y\'(x) for nodes on given range')

            def derivative_generator():
                dydx = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
                for t in t_range:
                    yield dydx(t)

            return np.array(list(derivative_generator()))      

        @lru_cache(maxsize=5)
        def get_xdt_from_parameter(t_range: tuple) -> np.array:

            def xdt_generator():
                xdt = lambda t: C * (1 - np.cos(2*t))
                for t in t_range:
                    yield xdt(t)

            return np.array(list(xdt_generator()))


        @lru_cache(maxsize=5)
        def get_integrand(y_range: tuple, dy_range: tuple, xdt_range: tuple) -> np.array:
            log.debug(msg=f'Computes integrand values for grid')

            def integrand_generator():
                integrand = lambda i: np.sqrt((1 + dy_range[i]**2) / y_range[i]) * xdt_range[i]
                for i, _ in enumerate(xdt_range):
                    yield integrand(i)
            
            return np.array(list(integrand_generator()))


        C, T = get_constants()
        a = 1e-7
        b = T
        n = self.SETTINGS['N']['max']
        fx, fy = get_parametrized_funcs()


        X_NODES, Y_NODES, T_NODES = get_nodes_from_parameter(a=a, b=b, n=n, fy=fy, fx=fx)
        YdX_NODES = get_ydx_from_parameter(t_range=tuple(T_NODES))
        Xdt_NODES = get_xdt_from_parameter(t_range=tuple(T_NODES))
        INTEGRAND = get_integrand(y_range=tuple(Y_NODES), dy_range=tuple(YdX_NODES), xdt_range=tuple(Xdt_NODES))

        self.T_NODES, self.INTEGRAND = T_NODES, INTEGRAND
        self.C, self.T = C, T

        log.info(msg=f'Model is set up')
        log.info(msg=f'Collected parameters:\n\
            X nodes:\t{X_NODES} ({len(X_NODES)} items)\n\
            Y nodes:\t{Y_NODES} ({len(Y_NODES)} items)\n\
            Ydx nodes:\t{YdX_NODES} ({len(YdX_NODES)} items)\n\
            T nodes:\t{T_NODES} ({len(T_NODES)} items)\n\
            Integrand:\t{INTEGRAND} ({len(INTEGRAND)} items)')
    
    def compare_methods(self) -> tuple:
        log.info(msg=f'Compares trapezoid and Simpson methods and logs results')
        
        T_NODES, INTEGRAND = self.T_NODES, self.INTEGRAND
        C, T = self.C, self.T
        n = self.SETTINGS['N']['max']
        G = self.SETTINGS['G']
        
        def select_n(x, y, n: int) -> tuple:
            if n < 1: raise ValueError
            h = (x[-1] - x[0]) / (n - 1)
            log.debug(msg=f'h = {h}')
            log.debug(msg=f'selecting {n} items')
            y_selected = []

            def n_items_generator():
                for i in range(n):
                    # log.debug(msg=f'i = {i} looks for {x[0] + h * i}')
                    nearest, i_nearest = find_nearest(x, x[0] + h * i)
                    # log.debug(msg=f'Nearest is {nearest}')
                    y_selected.append(y[i_nearest])
                    yield nearest

            x_selected = np.array(list(n_items_generator()))
            y_selected = np.asarray(y_selected)
            return x_selected, y_selected


        def find_nearest(array: np.array, value: float) -> tuple:
            idx = np.searchsorted(array, value, side='left')
            if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
                return array[idx - 1], idx - 1
            else:
                return array[idx], idx

        errors_trapezoid = []
        errors_simpson = []
        n_values = []

        reference = np.sqrt(2 * C / G) * T
        functional = lambda x: x / np.sqrt(2 * G)
        get_error = lambda x : np.abs( x - reference )

        log.info(msg=f'Reference value is: {reference:e}')

        absciasses_logspace = np.logspace(1, 4, 100, dtype=int)

        for nodes in absciasses_logspace:
            x_selected, y_selected = select_n(T_NODES, INTEGRAND, n=nodes)
            
            step = (x_selected[-1] - x_selected[0]) / (nodes - 1)
            log.debug(msg=f'Step is {step:e}')
            n_values.append(step)

            functional_value = functional( composite_trapezoid_ranged(x_selected, y_selected, n=nodes) )
            errors_trapezoid.append( get_error(functional_value) )
            log.debug(msg=f'Trapezoid error is {errors_trapezoid[-1]:e}')
            log.info(msg=f'min F[y] computed for {nodes} nodes (trapezoid): { functional_value :e}')
            
            functional_value = functional( composite_simpson_ranged(x_selected, y_selected, n=nodes) )
            errors_simpson.append( get_error(functional_value) )
            log.debug(msg=f'Simpson error is {errors_simpson[-1]:e}')
            log.info(msg=f'min F[y] computed for {nodes} nodes (simpson): { functional_value :e}')
        
        log.info(msg=f'Comparesison finished successfully, now plotting')
        return n_values, errors_simpson, errors_trapezoid


    def plot_log_errors(self, absciasses, simpson_error, trapeziod_error):

        log.info(msg=f'Plots errors and saves results')
        from .plotting import PlotArtist
        Artist = PlotArtist()
        Artist.add_log_plot(absciasses, trapeziod_error, style={
        'legend':'Trapezoid error',
        'color':'#67CC8E',
        'linestyle':'-'
        })
        Artist.add_log_plot(absciasses, simpson_error, style={
        'legend':'Simpson error',
        'color':'#9250BC',
        'linestyle':'-'
        })
        
        Artist.save_as('res/plots/integral')
        log.debug(msg=f'Saved figure of errors ')
        log.info(msg=f'Modeling finished. Plot is saved at "res/plots/integral.svg"')
