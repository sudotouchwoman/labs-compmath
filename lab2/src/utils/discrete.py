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
from .methods.trapezoid import composite_trapezoid
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

    def create_surfaces(self, dtype=np.float64) -> tuple:

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
        t_nodes = np.linspace(a, b, n_generated_nodes, dtype=dtype)
        log.debug(msg=f'T nodes: {t_nodes}')

        # collect t nodes and define integrand as function of t 
        # looking back, I facepalm myself for creating separate node collections for x(t), y(t), y'(x)
        # like, I could just find the int(t) explicitly from the very beginning
        # nevertheless, I would not refactor that pt now, too much time spent on that

        xdt = lambda t: C * (1 - np.cos(2*t))
        dydx = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
        y = lambda t: C * (0.5 - 0.5 * np.cos(2*t))

        integrand = lambda t: np.sqrt( ( 1 + dydx(t)**2 ) / y(t) / 2 / G ) * xdt(t)
        integrand_nodes = integrand(t_nodes)

        # compute the true functional minima (from analytical formulae)
        # define lambda for absolute error
        reference = np.sqrt(2 * C / G) * (b - a)
        log.info(msg=f'Nodes collected, reference value is {reference:e}')
        abs_error = lambda x: np.abs( x - reference )

        # init array for surfaces of the given shape
        # (2 as there are 2 surfaces, simpson and trapezoid)
        surfaces = np.zeros(shape=(ipl_items, int_items, 2))
        log.info(msg=f'Building the surfaces: the shape is {surfaces.shape}')

        # fill the array iteratively,
        # each cell represents error for quadrature on given integration nodes number
        # and interpolation nodes number
        for i, ipl_nodes in enumerate(interpolation_nodes_space):
            for j, int_nodes in enumerate(integration_nodes_space):
                approximator = DiscreteModel(x_nodes=t_nodes, y_nodes=integrand_nodes, n_int=int_nodes, n_ipl=ipl_nodes)
                surfaces[i, j] = abs_error(approximator.approximate())

        log.info(msg=f'Succesfully finished modeling')
        return surfaces, integration_nodes_space, interpolation_nodes_space

    def draw_surfaces(self, surfaces: np.ndarray, integration_nodes_space: np.ndarray, interpolation_nodes_space: np.ndarray):

        # use plotly to create great interactive surface 3d plot
        # from data collected in the previous method 
        # (use log scale for each axis)

        log.info(msg=f'Creates surface plot')
        fig = go.Figure(data=[
            go.Surface(
                name="Simpson",
                z=surfaces[:,:,0],
                x=integration_nodes_space,
                y=interpolation_nodes_space,
                opacity=0.8,
                hovertemplate=
                '<b>Absolute error (Simpson): %{z:e}</b>'+
                '<br>Integration nodes: %{x}'+
                '<br>Interpolation nodes: %{y}'+
                '<extra></extra>',
                colorscale='inferno',
                showscale=False),
            go.Surface(
                name="Trapezoid",
                z=surfaces[:,:,1],
                x=integration_nodes_space,
                y=interpolation_nodes_space,
                opacity=0.4,
                hovertemplate=
                '<b>Absolute error (Trapezoid): %{z:e}</b>'+
                '<br>Integration nodes: %{x}'+
                '<br>Interpolation nodes: %{y}'+
                '<extra></extra>',
                colorscale='magenta',
                showscale=False)
                ])
            
        fig.update_traces(
            contours_z=dict(show=False, project_z=False),
            contours_x=dict(show=True, highlightcolor='white', project_x=False),
            contours_y=dict(show=True, highlightcolor='white', project_y=False),)

        fig.update_layout(
            title='Absolute error surface',
            scene=dict(
                xaxis = dict(type='log' , nticks=10, range=[-4, 0],),
                yaxis = dict(type='log' , nticks=10, range=[-4, 0],),
                zaxis = dict(type='log',),
                xaxis_title='Integration nodes',
                yaxis_title='Interpolation nodes',
                zaxis_title='Absolute error'),
                width=1500,
                height=900)

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
        n_ipl = 5
        nodes = 1000

        # collect 1e3 nodes and select only n_ipl of them to fit the linear model
        # basically, I could just plot the selected nodes (this should yield the same result),
        # but here the very 'prediction' of the linear model was used
        fx, fy, *_ = BrachistochroneNodeProvider.get_parametrized_funcs(C=C)
        x_nodes, y_nodes, t_range = BrachistochroneNodeProvider.get_nodes_from_parameter(a, b, nodes, fy=fy, fx=fx)
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(x_nodes, y_nodes, n_ipl)
        ydx_nodes = BrachistochroneNodeProvider.get_ydx_from_parameter(t_range)

        model = LinearMethod().fit(x_nodes=x_selected, y_nodes=y_selected)

        x_model = np.arange(x_nodes[0], x_nodes[-1], step=0.01)
        y_model = np.array([model.predict_y(x) for x in x_model])
        ydx_model = np.array([model.predict_ydx(x) for x in x_model])

        log.debug(msg=f'Computed, now plotting')

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #     x=x_model,
        #     y=y_model,
        #     mode='markers',
        #     name=f'Linear (n = {n_ipl})'))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_nodes[1:],
            y=np.diff(x_nodes),
            mode='markers',
            name=f'X diff (n = {n_ipl})'))

        # fig.add_trace(go.Scatter(
        #     x=x_nodes,
        #     y=y_nodes,
        #     mode='lines',
        #     name=f'Brachistochrone'))

        # fig.add_trace(go.Scatter(
        #     x=x_model,
        #     y=ydx_model,
        #     mode='lines',
        #     name=f'Approx derivative (n = {n_ipl})'))

        # fig.add_trace(go.Scatter(
        #     x=x_nodes,
        #     y=ydx_nodes,
        #     mode='lines',
        #     name=f'True derivative'))

        fig.update_layout(
            title='X nodes diff (from np.diff)',
            autosize=True,
            width=1000,
            height=900)
        # fig.update_yaxes(range = [-0.5,1.5])
        fig.write_html("res/plots/diff.html")
        log.info(msg=f'Test finished')


class DiscreteModel:
    def __init__(self, x_nodes, y_nodes, n_int: int, n_ipl: int) -> None:
        self.X_NODES = np.asarray(x_nodes)
        self.Y_NODES = np.asarray(y_nodes)
        self.N_int = n_int
        self.N_ipl = n_ipl

    def approximate(self) -> float:
        # get the approximation of functional 
        # for provided configuration of interpolation and integration nodes
        X_NODES, Y_NODES = self.X_NODES, self.Y_NODES
        x_selected, y_selected = BrachistochroneNodeProvider.select_n(X_NODES, Y_NODES, self.N_ipl)
        model = LinearMethod().fit(x_selected, y_selected)

        @lru_cache(maxsize=1)
        def integrand(t):
            return model.predict_y(t)

        simpson = composite_simpson(x_selected[0], x_selected[-1], self.N_int, integrand)
        trapezoid = composite_trapezoid(x_selected[0], x_selected[-1], self.N_int, integrand)
        return (simpson, trapezoid)
        