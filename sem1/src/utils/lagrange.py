import matplotlib.pyplot as plt
import numpy as np
import math

canvas = plt.figure(figsize=(10, 10), dpi=300)
ax = canvas.add_subplot(111)

def plot_from_arrays(x: np.ndarray, *fx) -> None:
    STYLE = {
    'color':['#168039','#a24cc2', '#ffbe43', '#ff5252'],
    'linestyle':['-','--'],
    'legend':['$f(X)$', '$L_{2}(X)$'],
    'xlabel':'$\it{X}$',
    'ylabel':'$\it{f(X)}$'
}

    for i, f in enumerate(fx):
        ax.plot(x, f, color=STYLE['color'][i], linestyle=STYLE['linestyle'][i])

    plt.xlabel(STYLE['xlabel'])
    plt.ylabel(STYLE['ylabel'])
    plt.legend(STYLE['legend'], loc='best')
    plt.show()

def save_results(filename: str = "lagrange") -> None:
    canvas.savefig(filename+'.svg', format='svg')

def L2_interpolation(x1y1: tuple, x2y2: tuple, x3y3: tuple, X: np.ndarray) -> np.ndarray:
    xi = [point[0] for point in (x1y1, x2y2, x3y3)]
    yi = [point[1] for point in (x1y1, x2y2, x3y3)]

    L2 = lambda x: \
        yi[0]*((x - xi[1])*(x - xi[2])) / ((xi[0] - xi[1])*(xi[0] - xi[2])) + \
        yi[1]*((x - xi[0])*(x - xi[2])) / ((xi[1] - xi[0])*(xi[1] - xi[2])) + \
        yi[2]*((x - xi[0])*(x - xi[1])) / ((xi[2] - xi[1])*(xi[2] - xi[1]))

    lagrangian = np.zeros_like(X)
    for i, x_i in enumerate(X):
        lagrangian[i] = L2(x_i)

    return lagrangian

def interpolate(stepsize: float = 0.001, dtype = np.float32) -> None:
    X = np.arange(1.001, step=stepsize, dtype=dtype)
    fx = lambda x: \
        math.sqrt(x - x**2)
    
    Fx = np.array([fx(xi) for xi in X], dtype=dtype)
    points = [(xi, fx(xi)) for xi in (0, ((3+ math.sqrt(5))/6), 1)]

    lagrangian = L2_interpolation( *points, X=X )
    plot_from_arrays(X, Fx, lagrangian)

    for point in points:
        ax.plot(point[0], point[1], 'go')
    
    plot_error()


def plot_error() -> None:
    ax.vlines(
        x=[0.5],
        ymin=0.5, ymax=0.5+0.25,
        color='#ff5252', linestyle=':')

    ax.plot(0.5, 0.5, 'ro')
    ax.plot(0.5, 0.5+0.25, 'ro')