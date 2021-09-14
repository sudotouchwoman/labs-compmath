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
        self.AX.grid(linestyle=':')
        self.AX.set_ylim(0, 10)

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
        self.CANVAS.savefig(f'{filename}.{format}', format=format)

    def add_plot(self, x: np.ndarray, y:np.ndarray, style:dict = DEFSTYLE) -> None:
        self.AX.plot(x, y, linestyle=style.get('linestyle','-'), color=style.get('color', '#5793FF'))
