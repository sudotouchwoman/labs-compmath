'''
This module provides interface for advanced part of the problem

Especially the one for discretization of the brachistochrone
by linear interpolation and composite simpson method

'''
import numpy as np
import plotly.graph_objects as go
from functools import lru_cache
import logging
import os

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
        self.config = load_boundary_conds(filepath)
        log.debug(msg=f'Config loaded')

    def create_surface(self, dtype=np.float64) -> tuple:

        log.info(msg=f'Creates surface of error')

        # collect used conditions: 
        # min and max number of interpolation and integration nodes,
        # amount of items in the linspace 
        # (this is the shape of the error surface)
        ipl_min, ipl_max, ipl_items = (
            self.config['interpolation-nodes']['min'],
            self.config['interpolation-nodes']['max'],
            self.config['interpolation-nodes']['items'],
            )

        int_min, int_max, int_items = (
            self.config['integration-nodes']['min'],
            self.config['integration-nodes']['max'],
            self.config['integration-nodes']['items'],
            )

        log.debug(msg=f'Collected interpolation and integration settings')

        # at first I used linspace, but then I found out the geomspace 
        # which is basically logspace but the endpoints are specified excplicitly, not through exp notation
        # cast to int as these represent numbers of nodes

        interpolation_nodes_space = np.geomspace(ipl_min + 1, ipl_max + 1, ipl_items, dtype=int)
        integration_nodes_space = np.geomspace(int_min + 1, int_max + 1, int_items, dtype=int)

        log.debug(msg=f'Interpolation logspace: {interpolation_nodes_space}')
        log.debug(msg=f'Integration logspace: {integration_nodes_space}')

        # unpack constants for shorter var names
        # a, b - bounds for t
        # C, T - constants computed from boundary condition
        # n_generated_items - total number of nodes 
        # (must be greater or equal to the max number of interpolation nodes, actually the latter)
        G = self.config['G']
        C, T = self.config['C'], self.config['T']
        a = dtype(self.config['t0'])
        b = T
        n_generated_nodes = self.config['items']

        log.info(msg=f'Producing nodes')
        fx, fy = BrachistochroneNodeProvider.get_parametrized_funcs(C=C)
        x_nodes, y_nodes, _ = BrachistochroneNodeProvider.get_nodes_from_parameter(a, b, n_generated_nodes, fy=fy, fx=fx)

        log.debug(msg=f'X nodes: {x_nodes}')
        log.debug(msg=f'Y nodes: {y_nodes}')

        # compute the true functional minima (from analytical formulae)
        # define lambda for absolute error
        reference = np.sqrt(2 * C / G) * (T - a)
        log.info(msg=f'Nodes collected, reference value is {reference:e}')
        abs_error = lambda x: np.abs( x - reference )

        # init array for surface of the given shape
        surface = np.zeros(shape=(ipl_items, int_items))
        log.info(msg=f'Building the surface: the shape is {surface.shape}')

        # fill the array iteratively,
        # each cell represents error for quadrature on given integration nodes number
        # and interpolation nodes number
        for i, ipl_nodes in enumerate(interpolation_nodes_space):
            for j, int_nodes in enumerate(integration_nodes_space):
                approximator = DiscreteModel(x_nodes=x_nodes, y_nodes=y_nodes, n_int=int_nodes, n_ipl=ipl_nodes, G=G)
                surface[i, j] = abs_error(approximator.approximate())
                log.debug(msg=f'Error for {ipl_nodes} interpolation nodes and {int_nodes} integration nodes is {surface[i, j]:e}')

        log.info(msg=f'Succesfully finished modeling')
        return surface, integration_nodes_space, interpolation_nodes_space

    def draw_surface(self, surface: np.ndarray, integration_nodes_space: np.ndarray, interpolation_nodes_space: np.ndarray):

        # use plotly to create great interactive surface 3d plot
        # from data collected in the previous method 
        # (use log scale for each axis)
        log.info(msg=f'Creates surface plot')
        fig = go.Figure(data=[go.Surface(
            z=(surface),
            x=(1 / integration_nodes_space),
            y=(1 / interpolation_nodes_space),
            opacity=0.5,
            hovertemplate=
            '<b>Absolute error: %{z:e}</b>'+
            '<br>Integration step: %{x:.4f}'+
            '<br>Interpolation step: %{y:.4f}'+
            '<extra></extra>',
            colorscale='Plotly3',
            showscale=False)])
            
        fig.update_traces(
            contours_z=dict(show=False, project_z=False),
            contours_x=dict(show=True, highlightcolor='lightgreen', project_x=False),
            contours_y=dict(show=True, highlightcolor='lightgreen', project_y=False),)
        fig.update_layout(
            title='Absolute error surface',
            scene=dict(
                xaxis = dict(type='log' , nticks=10, range=[-4, 0],),
                yaxis = dict(type='log' , nticks=10, range=[-4, 0],),
                zaxis = dict(type='log' , nticks=20),
                xaxis_title='Integration step',
                yaxis_title='Interpolation step',
                zaxis_title='Absolute error of approximation (log)'),
                width=1500,
                height=900)

        fig.update_coloraxes(
            cmin=surface.min(),
            cmax=surface.max(),
            colorscale='Viridis')

        # plotly makes use of html and js for interactivity
        fig.write_html("res/plots/surface.html")
        log.info(msg=f'Surface plot saved')

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
        C, T = self.config['C'], self.config['T']
        a = self.config['t0']
        b = T
        n_int = 7
        n_ipl = 50
        nodes = 1000

        # collect 1e3 nodes and select only n_ipl of them to fit the linear model
        # basically, I could just plot the selected nodes (this should yield the same result),
        # but here the very 'prediction' of the linear model was used
        fx, fy = BrachistochroneNodeProvider.get_parametrized_funcs(C=C)
        x_nodes, y_nodes, t_range = BrachistochroneNodeProvider.get_nodes_from_parameter(a, b, nodes, fy=fy, fx=fx)
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(x_nodes, y_nodes, n_ipl)
        ydx_nodes = BrachistochroneNodeProvider.get_ydx_from_parameter(t_range)

        model = LinearMethod().fit(x_nodes=x_selected, y_nodes=y_selected)

        x_model = np.arange(x_nodes[0], x_nodes[-1], step=0.01)
        y_model = np.array([model.predict_y(x) for x in x_model])
        ydx_model = np.array([model.predict_ydx(x) for x in x_model])

        log.debug(msg=f'Computed, now plotting')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_model,
            y=y_model,
            mode='lines',
            name=f'Linear (n = {n_ipl})'))

        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='lines',
            name=f'Brachistochrone'))

        fig.add_trace(go.Scatter(
            x=x_model,
            y=ydx_model,
            mode='lines',
            name=f'Approx derivative (n = {n_ipl})'))

        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=ydx_nodes,
            mode='lines',
            name=f'True derivative'))

        fig.update_layout(
            title='Brachistochrone interpolation comparison',
            autosize=True,
            width=1000,
            height=900)
        fig.update_yaxes(range = [-0.5,1.5])
        fig.write_html("res/plots/interpolant.html")
        log.info(msg=f'Test finished')


class DiscreteModel:
    def __init__(self, x_nodes, y_nodes, n_int: int, n_ipl: int, G=9.8) -> None:
        self.X_NODES = np.asarray(x_nodes)
        self.Y_NODES = np.asarray(y_nodes)
        self.N_int = n_int
        self.N_ipl = n_ipl
        self.G = G

    def approximate(self, method=LinearMethod) -> float:
        # get the approximation of functional 
        # for provided configuration of interpolation and integration nodes
        X_NODES, Y_NODES = self.X_NODES, self.Y_NODES
        G = self.G
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(X_NODES, Y_NODES, self.N_ipl)
        model = method().fit(x_selected, y_selected)

        @lru_cache(maxsize=4)
        def integrand(x):
            # use cache as within the simpson quadrature 
            # the function is repeatedly applied to the current x node
            # (4 times per even and 2 times per odd one)
            result = np.sqrt( (1 + (model.predict_ydx(x))**2) / model.predict_y(x) )
            # log.debug(msg=f'y(x) prediction for {x} is {model.predict_y(x)}')
            # log.debug(msg=f'y\'(x) prediction for {x} is {model.predict_ydx(x)}')
            # log.debug(msg=f'Integrand for {x} is {result}')
            return result

        functional = lambda x: x / np.sqrt( 2 * G )

        approximation = composite_simpson(x_selected[0], x_selected[-1], self.N_int, integrand)
        return functional(approximation)
        