'''
Brachistochrone routines

`BrachistochroneApproximator` class wraps model creation and methods comparison
'''
import logging
import os
import numpy as np

from .methods.simpson import composite_simpson_ranged
from .methods.trapezoid import composite_trapezoid_ranged
from .optimizer import find_constants
from . import load_boundary_conds

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

class BrachistochroneErrorComputer:
    SETTINGS = None

    def __init__(self, filepath: str) -> None:
        log.debug(msg=f'Tries to load config from "{filepath}"')
        self.SETTINGS = load_boundary_conds(filepath)
        self.NODE_PROVIDER = NodeProvider()
        log.debug(msg=f'Config loaded')

    def get_parametrized_funcs(self, C):
        log.debug(msg=f'Sets functions for x and y for parameter t')
        funcs = (
            lambda t: C * (t - 0.5 * np.sin(2*t)),
            lambda t: C * (0.5 - 0.5 * np.cos(2*t))
        )
        return funcs

    def set_model(self, dtype = np.float64):
        log.info(msg=f'Setting up model')

        x_a, y_a = self.SETTINGS['Boundary-condition']['x'], self.SETTINGS['Boundary-condition']['y']

        C, T = self.NODE_PROVIDER.get_constants(x_a, y_a)
        a = dtype(1e-7)
        b = dtype(T)
        n = self.SETTINGS['N']['max']
        fx, fy = self.get_parametrized_funcs(C)


        X_NODES, Y_NODES, T_NODES = self.NODE_PROVIDER.get_nodes_from_parameter(a=a, b=b, n=n, fy=fy, fx=fx)
        YdX_NODES = self.NODE_PROVIDER.get_ydx_from_parameter(t_range=tuple(T_NODES))
        Xdt_NODES = self.NODE_PROVIDER.get_xdt_from_parameter(t_range=tuple(T_NODES))
        INTEGRAND = self.NODE_PROVIDER.get_integrand(y_range=tuple(Y_NODES), dy_range=tuple(YdX_NODES), xdt_range=tuple(Xdt_NODES))

        self.T_NODES, self.INTEGRAND = T_NODES, INTEGRAND
        self.C, self.T = C, T

        log.info(msg=f'Model is set up')
        log.info(msg=f'Collected parameters:\n\
            X nodes:\t{X_NODES} ({len(X_NODES)} items)\n\
            Y nodes:\t{Y_NODES} ({len(Y_NODES)} items)\n\
            Ydx nodes:\t{YdX_NODES} ({len(YdX_NODES)} items)\n\
            T nodes:\t{T_NODES} ({len(T_NODES)} items)\n\
            Integrand:\t{INTEGRAND} ({len(INTEGRAND)} items)')
    
    def compare_methods(self, dtype=np.float64) -> tuple:
        log.info(msg=f'Compares trapezoid and Simpson methods and logs results')
        
        T_NODES, INTEGRAND = self.T_NODES, self.INTEGRAND
        C, T = self.C, self.T
        n = self.SETTINGS['N']['max']
        G = dtype(self.SETTINGS['G'])
        
        def select_n(x, y, n: int) -> tuple:
            if n < 1: raise ValueError
            h = (x[-1] - x[0]) / (n - 1)
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
        # reference is computed from analytical formulae
        # at last I started using 1e-7 as lower bound
        # thus fixed error plots
        reference = np.sqrt(2 * C / G) * (T - 1e-7)
        functional = lambda x: x / np.sqrt(2 * G)
        get_error = lambda x : np.abs( x - reference )

        log.info(msg=f'Reference value is: {reference:e}')
        
        logspace_settings = self.SETTINGS['logspace']
        absciasses_logspace = np.logspace(
            logspace_settings['min'],
            logspace_settings['max'],
            logspace_settings['items'],
            dtype=int)
        # absciasses_logspace = np.arange(11, 9999 + 100, 100, dtype=int)

        for nodes in absciasses_logspace:
            x_selected, y_selected = select_n(T_NODES, INTEGRAND, n=nodes)
            
            step = (x_selected[-1] - x_selected[0]) / (nodes - 1)
            n_values.append(step)

            functional_value = functional( composite_trapezoid_ranged(x_selected, y_selected, n=nodes) )
            errors_trapezoid.append( get_error(functional_value) )
            
            functional_value = functional( composite_simpson_ranged(x_selected, y_selected, n=nodes) )
            errors_simpson.append( get_error(functional_value) )
        
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
        
        Artist.save_as('res/plots/absolute-error')
        log.debug(msg=f'Saved figure of errors ')
        log.info(msg=f'Modeling finished. Plot is saved at "res/plots/absolute-error.svg"')

class NodeProvider:
    def __init__(self) -> None:
        pass

    def get_constants(self, x_a, y_a):
            log.debug(msg=f'Computes arbitrary constants of model (C and T)')

            log.debug(msg=f'The endpoint is {x_a}{y_a}')

            C, T = find_constants((x_a, y_a))
            log.info(msg=f'Found constants: C = {C}, T = {T}')
            self.C, self.T = C, T
            return C, T

    def set_constants(self, C, T):
        if C is None: raise ValueError
        if T is None: raise ValueError
        self.C, self.T = C, T

    def get_nodes_from_parameter(self, a, b, n: int, fy, fx) -> tuple:
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

    def get_ydx_from_parameter(self, t_range: tuple):
        log.debug(msg=f'Computes y\'(x) for nodes on given range')

        def derivative_generator():
            dydx = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
            for t in t_range:
                yield dydx(t)

        return np.array(list(derivative_generator()))      

    def get_xdt_from_parameter(self, t_range: tuple) -> np.array:

        def xdt_generator():
            C = self.C
            xdt = lambda t: C * (1 - np.cos(2*t))
            for t in t_range:
                yield xdt(t)

        return np.array(list(xdt_generator()))


    def get_integrand(self, y_range: tuple, dy_range: tuple, xdt_range: tuple) -> np.array:
        log.debug(msg=f'Computes integrand values for grid')

        def integrand_generator():
            integrand = lambda i: np.sqrt((1 + dy_range[i]**2) / y_range[i]) * xdt_range[i]
            for i, _ in enumerate(xdt_range):
                yield integrand(i)
        
        return np.array(list(integrand_generator()))