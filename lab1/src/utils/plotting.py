import numpy as np
import matplotlib.pyplot as plt

canvas = plt.figure(figsize=(10, 10), dpi=300)
ax = canvas.add_subplot(111)

def plot_from_arrays(x: np.ndarray, *fx) -> None:
    '''
        Draw plots for given list of value-lists

        + `x`: np.array or list
            array of used x axis

        + `*fx`: list of arrays, or 2d np.array
            each corresponding to a certain function
    '''

    for i, f in enumerate(fx):
        ax.plot(x, f, linestyle=':')

    # plt.xlabel(STYLE['xlabel'])
    # plt.ylabel(STYLE['ylabel'])
    # plt.legend(STYLE['legend'], loc='best')
    plt.show()