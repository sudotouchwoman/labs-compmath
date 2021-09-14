import numpy as np
import logging
import os

log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'res/computations.log', encoding='utf-8')
formatter = logging.Formatter('[%(levelname)s]::[%(name)s]::%(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

from .methods import lagrange
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
    COEFFS = None
    ARTIST = None

    def __init__(self, x_nodes, y_nodes) -> None:
        self.X_NODES = x_nodes
        self.Y_NODES = y_nodes
        self.ARTIST = PlotArtist()
        self.COEFFS = self.get_coeffs()

    def get_coeffs(self):
        pass

    def compute_at_x(self, x):
        pass

    def compute_for_range(self, x_range):
        pass

    def print_coeffs() -> None:
        pass

    def plot_results(self) -> None:
        self.ARTIST.plot_points(self.X_NODES, self.Y_NODES)

class CubSqplineMethod(InterpolationMethod):

    def __init__(self, x_nodes, y_nodes) -> None:
        super().__init__(x_nodes, y_nodes)
    
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

    def plot_results(self, filename:str = 'splines', step:float = 0.001) -> None:
        super().plot_results()
        
        x_range = np.arange(self.X_NODES[-1] + step, step=step, dtype=np.float64)
        spline_values = self.compute_for_range(x_range=x_range)
        
        def print_coeffs(self) -> None:
            log.info(msg='Computed coefficients:')
            log.info(msg=f'A_i values vector:\n {self.Y_NODES}')
            log.info(msg=f'B_i values vector:\n {self.COEFFS[:, 0]}')
            log.info(msg=f'C_i values vector:\n {self.COEFFS[:, 1]}')
            log.info(msg=f'D_i values vector:\n {self.COEFFS[:, 2]}')

        print_coeffs(self)
        log.info(msg=f'Provided x range (using step {step}):\n{x_range}')
        log.info(msg=f'Corresponding spline values:\n{spline_values}')
        
        self.ARTIST.plot_from_arrays(x_range, spline_values)
        self.ARTIST.save_as(filename=filename)

class LagrangeMethod(InterpolationMethod):

    def __init__(self, x_nodes, y_nodes) -> None:
        super().__init__(x_nodes, y_nodes)
    pass