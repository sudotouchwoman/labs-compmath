import numpy as np
import logging
import os

LOGFILE = 'res/computations.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)
print(f'Writing log to {LOGFILE}')

from .methods.lagrange import L_at_x
from .methods import cubspline as spline
from .plotting import PlotArtist

def load_nodes(source:str) -> tuple:
    import json
    '''
        Collect node data from json file

        Schema is `[{'X': float, 'H': float}]`
    '''
    with open(source, 'r') as nodes_file:
        NODES = json.loads(nodes_file.read())
    log.debug(msg=f'Collected nodes from {source}')
    x = [node['X'] for node in NODES]
    y = [node['H'] for node in NODES]
    return (x, y)

class InterpolationMethod:
    '''
        Base class for interpolation method

        recieves set of `x_nodes` and corresponding `y_values` and produces coeffs for implemented method

        can compute approximated value for a specific `x` given or for entire interpolation range
    '''
    X_NODES = None
    Y_NODES = None
    ARTIST = None

    def __init__(self, x_nodes, y_nodes, no_artist: bool = False) -> None:
        self.X_NODES = x_nodes
        self.Y_NODES = y_nodes
        if not no_artist: self.ARTIST = PlotArtist()

    def compute_at_x(self, x):
        pass

    def compute_for_range(self, x_range):
        pass


    def interpolate(self) -> None:
        self.ARTIST.plot_points(self.X_NODES, self.Y_NODES)

class CubSqplineMethod(InterpolationMethod):
    COEFFS = None

    def __init__(self, x_nodes, y_nodes, no_artist: bool = False) -> None:
        super().__init__(x_nodes, y_nodes, no_artist)
        self.COEFFS = self.get_coeffs()
    
    def get_coeffs(self):
        return spline.cubic_spline_coeff(self.X_NODES, self.Y_NODES)

    def compute_at_x(self, x):

        # find range provided x belongs to
        def get_range(x) -> int:
            i = 0
            for x_node in self.X_NODES:
                if x < x_node: i += 1
                else: break
            return i
        
        range_idx = get_range(x)

        # create shorter and more readable variable names
        x_i = self.X_NODES
        a = self.Y_NODES
        b = self.COEFFS[0]
        c = self.COEFFS[1]
        d = self.COEFFS[2]

        # lambda computes value for i-th spline at given x
        S_i = lambda x, i:\
            a[i] + b[i]*(x - x_i[i]) + c[i]*(x - x_i[i])**2 + d[i]*(x - x_i[i])**3

        return S_i(x=x, i=range_idx)

    def compute_for_range(self, x_range):
        # here we assume that provided range step is smaller 
        # than the distance between neighbouring interpolation nodes
        # (like, for god's sake, why would we interpolate otherwise?)

        # iterate through range, increasing current range if needed
        current_range = 0
        spline_values = np.zeros_like(x_range, dtype=np.float64)

        # create shorter and more readable variable names
        x_nodes = self.X_NODES
        a = self.Y_NODES
        b = self.COEFFS[:, 0]
        c = self.COEFFS[:, 1]
        d = self.COEFFS[:, 2]

        # lambda computes value for i-th spline range and given x
        S_i = lambda x, i:\
            a[i] + b[i]*(x - x_nodes[i]) + c[i]*(x - x_nodes[i])**2 + d[i]*(x - x_nodes[i])**3    
        
        for  idx, x_i in enumerate(x_range):
            if x_i > self.X_NODES[current_range + 1]: current_range += 1
            spline_values[idx] = S_i(x_i, current_range)

        return spline_values

    def compute_d_at_x(self, x):

        # find range provided x belongs to
        def get_range(x) -> int:
            i = 0
            for x_node in self.X_NODES:
                if x < x_node: i += 1
                else: break
            return i
        
        range_idx = get_range(x)

        # create shorter and more readable variable names
        x_i = self.X_NODES
        b = self.COEFFS[:, 0]
        c = self.COEFFS[:, 1]
        d = self.COEFFS[:, 2]

        # lambda computes 1st derivative value for i-th spline at given x
        d_S_i = lambda x, i:\
            b[i] + 2*c[i]*(x - x_i[i]) + 3**d[i]*(x - x_i[i])**2

        return d_S_i(x=x, i=range_idx)

    def compute_d_for_range(self, x_range):
        # here we assume that provided range step is smaller 
        # than the distance between neighbouring interpolation nodes
        # (like, for god's sake, why would we interpolate otherwise?)
        # i basically merged compute_d_at_x and compute_for_range here
        # the idea is that by evaluating result in a sequential way, there are less computations done when trying
        # to resolve range given x belongs to
        # (like, single condition instead of O(n) get_range() in methods for single x
        # it is not something one should bother when computing S(x) for single x, but for huge amounts of x points
        # this is something to be aware of and reason for small step of x_range, in order to avoid overlap)
        # I just don't want to burn my processor

        # iterate through range, increasing current range if needed
        current_range = 0
        d_spline_values = np.zeros_like(x_range, dtype=np.float64)

        # create shorter and more readable variable names
        x_i = self.X_NODES
        b = self.COEFFS[:, 0]
        c = self.COEFFS[:, 1]
        d = self.COEFFS[:, 2]

        # lambda computes 1st derivative value for i-th spline range and given x
        d_S_i = lambda x, i:\
            b[i] + 2*c[i]*(x - x_i[i]) + 3**d[i]*(x - x_i[i])**2
        
        for  idx, x_i in enumerate(x_range):
            if x_i > self.X_NODES[current_range]: current_range += 1
            d_spline_values[idx] = d_S_i(x_i, current_range)

        return d_spline_values

    def interpolate(self, filename:str = 'splines', step:float = 0.001) -> None:
        '''
            Compute cubic spline values with provided step, draw resulting curve and base nodes

            + `filename`: str
                where to save plot (`svg` format is used)
            
            + `step`: float
                increment, as points to compute interpolant are distributed uniformly
        '''
        log.info(msg=f'[{self.__class__.__name__}] Performs computations and saves results')
        x_range = np.arange(self.X_NODES[-1] + step, step=step, dtype=np.float64)
        spline_values = self.compute_for_range(x_range=x_range)
        
        def print_coeffs(self) -> None:
            log.debug(msg=f'[{self.__class__.__name__}] Computed coefficients:')
            log.debug(msg=f'[{self.__class__.__name__}] A_i values vector: {self.Y_NODES}')
            log.debug(msg=f'[{self.__class__.__name__}] B_i values vector: {self.COEFFS[:, 0]}')
            log.debug(msg=f'[{self.__class__.__name__}] C_i values vector: {self.COEFFS[:, 1]}')
            log.debug(msg=f'[{self.__class__.__name__}] D_i values vector: {self.COEFFS[:, 2]}')

        print_coeffs(self)
        log.info(msg=f'[{self.__class__.__name__}] Provided x range (using step {step}):{x_range}')
        log.info(msg=f'[{self.__class__.__name__}] Corresponding spline values:{spline_values}')
        style = {
            'legend':['$\it{S(x)}$'],
            'color':'#7B1FA2',
            'linestyle':'-'
        }
        self.ARTIST.plot_from_arrays(x_range, spline_values, style=style)
        super().interpolate()
        self.ARTIST.save_as(filename=filename)
        log.info(msg=f'[{self.__class__.__name__}] Created plot at {filename}')

class LagrangeMethod(InterpolationMethod):

    def __init__(self, x_nodes, y_nodes, no_artist:bool = False) -> None:
        super().__init__(x_nodes, y_nodes, no_artist)
    
    def compute_at_x(self, x):
        return L_at_x(x=x, x_nodes=self.X_NODES, y_nodes=self.Y_NODES)

    def compute_for_range(self, x_range):
        approximation = np.zeros_like(x_range)
        for i, x in enumerate(x_range):
            approximation[i] = self.compute_at_x(x)
        return approximation

    def interpolate(self, filename:str = 'lagrange', step:float = 0.001) -> None:
        '''
            Compute Lagrange polynom values with provided step, draw resulting curve and base nodes

            + `filename`: str
                where to save plot (`svg` format is used)
            
            + `step`: float
                increment, as points to compute interpolant are distributed uniformly
        '''
        log.info(msg=f'[{self.__class__.__name__}] Performs computations and saves results')
        x_range = np.arange(self.X_NODES[-1] + step, step=step, dtype=np.float64)
        L_x_range = self.compute_for_range(x_range=x_range)

        log.info(msg=f'[{self.__class__.__name__}] Provided x range (using step {step}):{x_range}')
        log.info(msg=f'[{self.__class__.__name__}] Corresponding Lagrange interpolant values:{L_x_range}')

        style = {
            'legend':[f'$\it{{L_{{{len(self.X_NODES)-1}}}(x)}}$'],
            'color':'#67CC8E',
            'linestyle':'-'
        }
        self.ARTIST.plot_from_arrays(x_range, L_x_range, style=style)
        super().interpolate()
        self.ARTIST.save_as(filename=filename)
        log.info(msg=f'[{self.__class__.__name__}] Created plot at {filename}')