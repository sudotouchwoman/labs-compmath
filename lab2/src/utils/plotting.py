'''
# Plotting routines

Module contains simple `PlotArtist` class I implemented for drawing plots (who could've guessed?)

Basically, we need plotting set of points and sequence of functions, `plot_from_arrays()` and `plot_points()` are doing just that
'''
import numpy as np
import matplotlib.pyplot as plt

DEFSTYLE = {
        'legend':['$\it{F(x)}$'],
        'color':'#67CC8E',
        'linestyle':'--'
    }

class PlotArtist:
    CANVAS = None
    AX = None
    def __init__(self, figsize:tuple = (10, 10), dpi:int = 300) -> None:
        self.CANVAS = plt.figure(figsize=figsize, dpi=dpi)
        self.AX = self.CANVAS.add_subplot(111)
        # I added following lines for same scale as Lagrange interpolation initially produced huge errors
        # and messed up entire plot. It just looks better in this way
        # also add dotted grid
        self.AX.grid(linestyle=':')
        # self.AX.set_ylim(1e-5, 1e7)
        self.AX.tick_params(axis='both', which='major', labelsize=20)
        self.AX.tick_params(axis='both', which='minor', labelsize=14)

    def plot_from_arrays(self, x: np.ndarray, *fx, style:dict = DEFSTYLE) -> None:
        '''
            Draw plots for given list of value-lists

            + `x`: np.array or list
                array of used x axis

            + `*fx`: list of arrays, or 2d np.array
                each corresponding to a certain function
        '''

        for _, f in enumerate(fx):
            self.AX.plot(x, f, linestyle=style.get('linestyle','-'), color=style.get('color', '#5793FF'))

        plt.legend(style['legend'], loc='best')

    def plot_points(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
            Draw points formed by given arrays

            + `x`: 1D array

            + `y`: 1D array, with same shape as `x`
        '''
        for point in zip(x,y):
            self.AX.plot(point[0], point[1], 'ro')

    def save_as(self, filename:str = 'results', format:str = 'svg') -> None:
        '''
            Save current state of figure with given filename and extention

            by default, figures are saved in `svg` format for better quality, but this property can be overridden
        '''
        self.AX.legend(loc='best')
        self.CANVAS.savefig(f'{filename}.{format}', format=format)

    def add_plot(self, x: np.ndarray, y:np.ndarray, style:dict = DEFSTYLE) -> None:
        '''
            Add single plot to existing figure
        '''
        self.AX.plot(x, y, linestyle=style.get('linestyle','-'), color=style.get('color', '#5793FF'))

    def add_log_plot(self, x: np.ndarray, y:np.ndarray, style:dict = DEFSTYLE) -> None:
        '''
            Add single plot to existing figure
        '''
        self.AX.scatter(x, y, linestyle=style.get('linestyle','-'), color=style.get('color', '#5793FF'), label=style.get('legend', 'F(x)'))
        self.AX.set_yscale('log')
        self.AX.set_xscale('log')

    def fill_between(self, x, *curves, style:dict = {'alpha': 0.5, 'color':'#c4c4c4'}):
        '''
            Color area between provided curves on range x 
        '''
        self.AX.fill_between(x, *curves, color=style['color'], alpha=style['alpha'])