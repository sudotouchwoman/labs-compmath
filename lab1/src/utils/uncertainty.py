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

from .methods.lagrange import L_at_x
from .interpolate import load_nodes, LagrangeMethod, CubSqplineMethod
from .plotting import PlotArtist

class SubsetTester:
    SETTINGS = None
    X_VECTORS = None
    Y_VECTORS = None
    CI = []
    AVG = []
    ARTIST = None
    POLYNOMS = []
    NODES = None

    def __init__(self, settings:dict) -> None:
        self.SETTINGS = settings
        self.ARTIST = PlotArtist()

    def produce_vectors(self) -> None:
        x, y = load_nodes(self.SETTINGS['NODES'])
        self.NODES = {'x': x, 'y': y}
        
        def generate_random_vectors(initial, n: int):
            SETINGS = self.SETTINGS
            for i in range(n):
                random_vector = np.array(initial)
                for i, _ in enumerate(random_vector):
                    random_vector[i] += np.random.normal(loc=SETINGS['MEAN'], scale=SETINGS['SD'])
                yield random_vector
        
        if self.SETTINGS['TARGET'] == 'X':
            self.X_VECTORS = np.array( [vec for vec in generate_random_vectors(x, self.SETTINGS['VECTORS'])] )
            self.Y_VECTORS = np.array(y)
        if self.SETTINGS['TARGET'] == 'H':
            self.X_VECTORS = np.array(x)
            self.Y_VECTORS = np.array( [vec for vec in generate_random_vectors(y, self.SETTINGS['VECTORS'])] )

    def produce_polynoms(self) -> None:
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']
        uncertain = self.SETTINGS['TARGET']
        
        if uncertain == 'X':
            for sample in self.X_VECTORS:
                x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
                #x_range = np.arange(start=sample[0], stop=sample[-1] + step, step=step)
                model = LagrangeMethod(sample, self.Y_VECTORS, no_artist=True)
                self.POLYNOMS.append( np.array(model.compute_for_range(x_range)) )
        
        if uncertain == 'H':
            x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)
            for sample in self.Y_VECTORS:
                model = LagrangeMethod(self.X_VECTORS, sample, no_artist=True)
                self.POLYNOMS.append( np.array(model.compute_for_range(x_range)) )

        self.POLYNOMS = np.array(self.POLYNOMS)

    def compute_CI(self) -> None:

        def CI_for_sample(sample, q:float) -> tuple:
            '''
                Compute confidence interval (CI) for given subset `sample` with confidence level `q`

                Return [L, U] tuple
            '''
            n = len(sample)
            sample_mean = np.mean(sample)
            T = stats.t.ppf(q=q, df=n-1)
            sigma = np.std(sample) / math.sqrt(len(sample))

            margin_of_err = T * sigma

            CI = (sample_mean - margin_of_err, sample_mean + margin_of_err)
            return CI

        total_points = self.POLYNOMS.shape[1]
        
        for i in range(total_points):
            self.CI.append( CI_for_sample(self.POLYNOMS[:,i], q=self.SETTINGS['q']) )
            self.AVG.append( np.mean(self.POLYNOMS[:,i]) )

        self.CI = np.array(self.CI)
        self.AVG = np.array(self.AVG)

    def draw_chaos(self) -> None:
        step = self.SETTINGS['STEP']
        limits = self.SETTINGS['BOUNDS']
        plotname = self.SETTINGS['PLOTNAME']
        vectors = self.SETTINGS['VECTORS']

        x_range = np.arange(start=limits[0], stop=limits[1] + step, step=step)

        for i in range(vectors):
            if i % 100 != 0: continue
            self.ARTIST.add_plot(x_range, self.POLYNOMS[i,:], {'color': '#91C46C', 'linestyle': ':'})
        
        self.ARTIST.add_plot(x_range, self.CI[:, 0], {'color': '#ff6d71', 'linestyle': '--'})
        self.ARTIST.add_plot(x_range, self.CI[:, 1], {'color': '#ff6d71', 'linestyle': '--'})
        self.ARTIST.add_plot(x_range, self.AVG, {'color': '#3a66c5', 'linestyle': '--'})
        self.ARTIST.plot_points(self.NODES['x'], self.NODES['y'])

        self.ARTIST.save_as(plotname)



def perform_analysis(config: dict) -> None:

    Tester = SubsetTester(config)
    Tester.produce_vectors()
    Tester.produce_polynoms()
    Tester.compute_CI()
    Tester.draw_chaos()


        
    
def load_config(config_path:str) -> dict:
    import json
    with open(config_path, 'r') as nodes_file:
        settings = json.loads(nodes_file.read())
    log.debug(msg=f'Collected settings from {config_path}')
    return settings