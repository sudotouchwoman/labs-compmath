import numpy as np
import scipy.stats as stats
import math
import logging
import os

LOGFILE = 'res/uncertainty.log'
log = logging.getLogger(__name__)
DEBUGLEVEL = os.getenv('DEBUG_LEVEL','DEBUG')
log.setLevel(getattr(logging, DEBUGLEVEL))
log.disabled = os.getenv('LOG_ON', "True") == "False"
handler = logging.FileHandler(filename=f'{LOGFILE}', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s]::[%(levelname)s]::[%(name)s]::%(message)s', '%H:%M:%S')
handler.setFormatter(formatter)
log.addHandler(handler)
print(f'Writing log to {LOGFILE}')

from .interpolate import load_nodes, LagrangeMethod, CubSplineMethod
from .plotting import PlotArtist

class SubsetTester:
    SETTINGS = None
    CI = None
    AVG = None
    POLYNOM_VALUES = None
    NODES = None

    def __init__(self, settings:dict) -> None:
        self.SETTINGS = settings
        x, y = load_nodes(settings['NODES'])
        self.NODES = {'x': x, 'y': y}
        self.CI = []
        self.AVG = []
        self.POLYNOM_VALUES = []
        self.ARTIST = PlotArtist()

    def produce_vectors(self) -> None:
        pass

    def produce_polynoms(self) -> None:
        pass

    def compute_CI(self) -> None:
        log.debug(msg=f'Computes confidence interval for each x value (using uniformly distributed values within {self.SETTINGS["BOUNDS"]}, step {self.SETTINGS["STEP"]}')

        # transpose polynom values for easier access
        # for each slice (corresponding to all polynom values at specific x) 
        # compute confidence interval as
        # [mean - T*sigma; mean + T*sigma], thanks to scipy
        for ith_slice in list(zip(*self.POLYNOM_VALUES)):
            self.CI.append( stats.norm.interval(self.SETTINGS['q'], loc=np.mean(ith_slice), scale=np.std(ith_slice)) )
            self.AVG.append( np.mean(ith_slice) )

    def draw_results(self) -> None:
        log.info(msg=f'Makes plots')

        ARTIST = PlotArtist()
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']

        # once again create array with all x values to plot for
        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)

        # also add original nodes (points) and plot CI along with averaged polynom (latter was formed by means)
        lower_bound = [bound[0] for bound in self.CI]
        upper_bound = [bound[1] for bound in self.CI]

        ARTIST.add_plot(x_range, lower_bound, {'color': '#ff6d71', 'linestyle': ':'})
        ARTIST.add_plot(x_range, upper_bound, {'color': '#ff6d71', 'linestyle': ':'})
        ARTIST.fill_between(x_range, lower_bound, upper_bound)
        ARTIST.add_plot(x_range, self.AVG, {'color': '#3a66c5', 'linestyle': '--'})
        ARTIST.plot_points(self.NODES['x'], self.NODES['y'])

        # plot several polynoms from subset
        for i, ith_slice in enumerate(self.POLYNOM_VALUES):
            if i % 100 != 0: continue
            ARTIST.add_plot(x_range, ith_slice, {'color': '#91C46C', 'linestyle': ':'})


        ARTIST.save_as(self.SETTINGS['PLOTNAME'])
        log.info(msg=f'Plots saved as {self.SETTINGS["PLOTNAME"]}')
        

        
    
def load_config(config_path:str) -> dict:
    '''
        Load `json` file from provided location and parse it into config `dict`
    '''
    import json
    with open(config_path, 'r') as nodes_file:
        settings = json.loads(nodes_file.read())
    log.debug(msg=f'Collected settings from {config_path}')
    return settings

class UncertainX(SubsetTester):
    X_VECTORS = None
    Y_VECTOR = None

    def produce_vectors(self) -> None:
        '''
            Create set of X vectors containing errors

            Error settings are gathered from `SETTINGS` attribute
        '''
        log.debug(msg=f'Generates random X vectors ({self.SETTINGS["VECTORS"]})')
        def generate_random_vectors(initial, n: int):
            SETINGS = self.SETTINGS
            for i in range(n):
                random_vector = np.array(initial)
                for i, _ in enumerate(random_vector):
                    random_vector[i] += np.random.normal(loc=SETINGS['MEAN'], scale=SETINGS['SD'])
                yield random_vector
        
        self.X_VECTORS = np.array( [vec for vec in generate_random_vectors(self.NODES['x'], self.SETTINGS['VECTORS'])] )
        self.Y_VECTOR = np.array(self.NODES['y'])        
    
    def produce_polynoms(self) -> None:
        '''
            Perform interpolation for each set of previously created nodes
        '''
        log.debug(msg=f'Generates interpolation results for each of {len(self.X_VECTORS)} random X vectors')
        
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']

        # create array of all values to compute for
        # note that for X case bounds are generally narrower
        # as interpolation ranges for each X instanse do vary actually
        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
        
        for sample in self.X_VECTORS:
            model = CubSplineMethod(sample, self.Y_VECTOR)
            self.POLYNOM_VALUES.append( model.compute_for_range(x_range) )
        

class UncertainH(SubsetTester):
    X_VECTOR = None
    Y_VECTORS = None

    def produce_vectors(self) -> None:    
        '''
            Create set of H(x) vectors containing errors

            Error settings are gathered from `SETTINGS` attribute
        '''
        log.debug(msg=f'Generates random Y vectors ({self.SETTINGS["VECTORS"]})')
        def generate_random_vectors(initial, n: int):
            SETINGS = self.SETTINGS
            for i in range(n):
                random_vector = np.array(initial)
                for i, _ in enumerate(random_vector):
                    random_vector[i] += np.random.normal(loc=SETINGS['MEAN'], scale=SETINGS['SD'])
                yield random_vector
        
        self.X_VECTOR = np.array(self.NODES['x'])
        self.Y_VECTORS = np.array( [vec for vec in generate_random_vectors(self.NODES['y'], self.SETTINGS['VECTORS'])] )
        
    
    def produce_polynoms(self) -> None:
        '''
            Perform interpolation for each set of previously created nodes
        '''
        log.debug(msg=f'Generates interpolation results for each of {len(self.Y_VECTORS)} random Y vectors')
        
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']
        
        # create array of all values to compute for
        # in this case (Y contains errors) interpolation range 
        # is the same as x are accurate (and, accordingly, do not vary)
        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
        
        for sample in self.Y_VECTORS:
                model = CubSplineMethod(self.X_VECTOR, sample)
                self.POLYNOM_VALUES.append( model.compute_for_range(x_range) )

def X_analysis(config: dict) -> None:
    # complete pipeline for the case when x values contain errors
    log.info(msg=f'New task: perform analysis using {UncertainX}')
    log.debug(msg=f'Usage: {config}')

    XTester = UncertainX(config)
    XTester.produce_vectors()
    XTester.produce_polynoms()
    XTester.compute_CI()
    XTester.draw_results()
    log.info(msg=f'Task finished')

def H_analysis(config: dict) -> None:
    # complete pipeline for the case when h(x) (may be referred as y or Y) values contain errors
    log.info(msg=f'New task: perform analysis using {UncertainH}')
    log.debug(msg=f'Usage: {config}')

    HTester = UncertainH(config)
    HTester.produce_vectors()
    HTester.produce_polynoms()
    HTester.compute_CI()
    HTester.draw_results()
    log.info(msg=f'Task finished')