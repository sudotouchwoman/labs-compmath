import numpy as np
import scipy.stats as stats
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

# choose which method to use and change its name accordingly in `produce_polynoms() method`
# shame on me, should have found out a way to pass this param
# (actually, it is possible to pass class name as an argument, but it seemed too messy
# then, it is possible to store class name in json and do eval()
# well, it is an option but sounds kinda cringe and insecure, one should better avoid eval() wherever possible
# that's why this feature was left as it is ¯\_(ツ)_/¯)

from .interpolate import load_nodes, LagrangeMethod, CubSplineMethod # import classes used for interpolation
from .plotting import PlotArtist # import plotting routines

def load_config(config_path:str) -> dict:
    '''
        Load `json` file from provided location and parse it into config `dict`
    '''
    import json
    with open(config_path, 'r') as nodes_file:
        settings = json.loads(nodes_file.read())
    log.debug(msg=f'Collected settings from {config_path}')
    return settings
    

class SubsetTester:
    '''
        Base class for testing consequenses of errors in input data

        1. Loads a bunch of settings (mu, sigma for Z, 
        number of random vectors, interpolation range bounds)

        1. Creates set of random vectors of needed size

        1. Interpolates each one using provided method

        1. Plots results

        See `X_analysis` and `Y_analysis` functions below, those 
    '''
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
        raise NotImplementedError

    def produce_polynoms(self) -> None:
        raise NotImplementedError

    def compute_CI(self) -> None:
        log.debug(msg=f'Computes confidence interval for each x value (using uniformly distributed values within {self.SETTINGS["BOUNDS"]}, step {self.SETTINGS["STEP"]}')

        # transpose polynom values for easier access
        # for each slice (corresponding to all polynom values at specific x) 
        # compute confidence interval as
        # [mean - T*sigma; mean + T*sigma], thanks to scipy
        for ith_slice in list(zip(*self.POLYNOM_VALUES)):
            self.CI.append( stats.norm.interval(self.SETTINGS['q'], loc=np.mean(ith_slice), scale=stats.sem(ith_slice)) )
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
        # uncomment to fill area between CI bounds
        #ARTIST.fill_between(x_range, lower_bound, upper_bound)
        ARTIST.add_plot(x_range, self.AVG, {'color': '#3a66c5', 'linestyle': '--'})
        ARTIST.plot_points(self.NODES['x'], self.NODES['y'])

        # plot several polynoms from subset
        # comment in case you wish to fill CI without messing everything up
        for i, ith_slice in enumerate(self.POLYNOM_VALUES):
            if i % 100 != 0: continue
            ARTIST.add_plot(x_range, ith_slice, {'color': '#91C46C', 'linestyle': ':'})


        ARTIST.save_as(self.SETTINGS['PLOTNAME'])
        log.info(msg=f'Plots saved as {self.SETTINGS["PLOTNAME"]}')
        

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
        log.debug(msg=f'Generated random vectors')
    
    def produce_polynoms(self) -> None:
        '''
            Perform interpolation for each set of previously created nodes
        '''
        log.debug(msg=f'Generates interpolation results for each of {len(self.X_VECTORS)} random X vectors')
        
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']

        # create array of all values to compute for
        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
        
        for sample in self.X_VECTORS:
            # change CubsplineMethod to LagrangeMethod, if needed
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
        log.debug(msg=f'Generated random vectors')
        
    
    def produce_polynoms(self) -> None:
        '''
            Perform interpolation for each set of previously created nodes
        '''
        log.debug(msg=f'Generates interpolation results for each of {len(self.Y_VECTORS)} random Y vectors')
        
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']
        
        # create array of all values to compute for
        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
        
        for sample in self.Y_VECTORS:
            # change CubsplineMethod below to LagrangeMethod, if needed
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
    # complete pipeline for the case when h(x) (may be referred to as y or Y) values contain errors
    log.info(msg=f'New task: perform analysis using {UncertainH}')
    log.debug(msg=f'Usage: {config}')

    HTester = UncertainH(config)
    HTester.produce_vectors()
    HTester.produce_polynoms()
    HTester.compute_CI()
    HTester.draw_results()
    log.info(msg=f'Task finished')