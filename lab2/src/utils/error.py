'''
Brachistochrone routines

`BrachistochroneApproximator` class wraps model creation and methods comparison
`BrachistochroneNodeProvider` contains static methods to produce node collections needed during the experiment
'''
import logging
import os
import numpy as np
from abc import ABC

from .methods.simpson import composite_simpson
from .methods.trapezoid import composite_trapezoid
from .optimizer import find_constants
from .methods.plotting import PlotArtist
from . import load_boundary_conds

LOGFILE = 'res/error.log'
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
    '''
    This class contains methods to create needed sets of nodes, pick specified number of latter,
    compute simpson and trapezoid quadratures and compare their absolute errors

    It utilizes `NodeProvider` class to easily produce certain nodes for brachistochrone integrand
    '''
    SETTINGS = None

    def __init__(self, filepath: str) -> None:
        # config contains such parameters as 
        # brachistochrone endpoint, node range to use in comparison, 
        # here it is assumed that 
        # the optimal function is known and is the parametrized cycloid 
        # (thus we need to compute C and T from the boundary condition)
        log.debug(msg=f'Tries to load config from "{filepath}"')
        self.SETTINGS = load_boundary_conds(filepath)
        log.debug(msg=f'Config loaded')


    def compare_methods(self, dtype=np.float64) -> tuple:
        '''
        this method 
        repeatedly computes quadratures and absolute error, 
        returns results to be later fed to `plot_errors_logscale`
        '''

        log.info(msg=f'Compares trapezoid and Simpson methods and logs results')

        boundary_condition = self.SETTINGS['Boundary-condition']
        x_a, y_a = boundary_condition['x'], boundary_condition['y']
        C, T = BrachistochroneNodeProvider.get_constants(x_a, y_a)
        a = dtype(boundary_condition['t0'])
        b = dtype(T)
        G = dtype(self.SETTINGS['G'])
        integrand_f = BrachistochroneNodeProvider.get_integrand_func(C=C)

        errors_trapezoid = []
        errors_simpson = []
        n_values = []
        # reference is computed from analytical formulae
        # at last I started using 1e-7 as lower bound
        # this fixed error plots and produced unavoidable error
        reference = np.sqrt(2 * C / G) * (b - a)
        functional = lambda x: x / np.sqrt(2 * G)
        get_error = lambda x : np.abs( x - reference )

        log.info(msg=f'Reference value is: {reference:e}')
        
        # create logspace array to finely plot on logscale
        absciasses_logspace = np.geomspace(
            self.SETTINGS['min'],
            self.SETTINGS['max'],
            self.SETTINGS['items'],
            dtype=int)

        for nodes in absciasses_logspace:
            # comparison loop: compute quadratures and append to corresponding lists
            # make use of lambdas defined above            
            step = (b - a) / (nodes - 1)
            n_values.append(step)

            functional_value = functional( composite_trapezoid(a, b, nodes, integrand_f) )
            errors_trapezoid.append( get_error(functional_value) )
            
            functional_value = functional( composite_simpson(a, b, nodes, integrand_f) )
            errors_simpson.append( get_error(functional_value) )
        
        log.info(msg=f'Comparison finished successfully, now plotting')
        return n_values, errors_simpson, errors_trapezoid


    def plot_errors_logscale(self, absciasses, simpson_error, trapeziod_error):
        # plot absolute error for given absciasses and ordinate errors
        # utilize `PlotArtist` class collected in `compare_methods`
        log.info(msg=f'Plots errors and saves results')
        
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


class BrachistochroneNodeProvider(ABC):

    @staticmethod
    def get_constants(x_a, y_a):
        log.debug(msg=f'Computes arbitrary constants of model (C and T)')
        log.debug(msg=f'The endpoint is ({x_a},{y_a})')

        C, T = find_constants((x_a, y_a))
        log.info(msg=f'Found constants: C = {C}, T = {T}')
        return C, T

    @staticmethod
    def get_parametrized_funcs(C):
        # the funcs tuple contains x(t), y(t), y'(x), x'(t) expressions
        # these are required to compute the integrand value for given t
        log.debug(msg=f'Sets functions for x and y for parameter t')
        funcs = (
            lambda t: C * (t - 0.5 * np.sin(2*t)),
            lambda t: C * (0.5 - 0.5 * np.cos(2*t)),
            lambda t: np.sin(2*t) / (1 - np.cos(2*t)),
            lambda t: C * (1 - np.cos(2*t))
        )
        return funcs

    @staticmethod
    def get_nodes_from_parameter(a, b, n: int, fy, fx) -> tuple:
        log.debug(msg=f'Fits nodes on given range [{a};{b}] ({n} nodes)')
        
        if a > b: raise ValueError
        if n < 2: raise ValueError
        if fy is None or fx is None: raise ValueError
        t_range = np.linspace(a, b, n)
        x_range = fx(t_range)
        y_range = fy(t_range)
        
        log.debug(msg=f'Node collections computed')
        return x_range, y_range, t_range

    @staticmethod
    def get_ydx_from_parameter(t_range: tuple):
        log.debug(msg=f'Computes y\'(x) for nodes on t grid')

        ydx_f = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
        ydx = ydx_f(t_range)

        return ydx

    @staticmethod
    def get_integrand_func(C):
        _, yt, ydx, xdt = BrachistochroneNodeProvider.get_parametrized_funcs(C=C)
        integrand_f = lambda t: np.sqrt((1 + ydx(t)**2) / yt(t)) * xdt(t)
        return integrand_f

    @staticmethod
    def find_nearest(array: np.array, value: float) -> tuple:
        idx = np.searchsorted(array, value, side='left')
        if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
            return array[idx - 1], idx - 1
        else:
            return array[idx], idx

    @staticmethod
    def select_n(x, y, n: int) -> tuple:
        if n < 1: raise ValueError
        h = (x[-1] - x[0]) / (n - 1)

        y_selected = np.zeros(n)
        x_selected = np.zeros(n)

        for i in range(n):
            _, i_nearest = BrachistochroneNodeProvider.find_nearest(x, x[0] + h * i)
            y_selected[i] = y[i_nearest]
            x_selected[i] = x[i_nearest]

        return x_selected, y_selected