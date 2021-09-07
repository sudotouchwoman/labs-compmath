import matplotlib.pyplot as plt
import numpy as np
import math

canvas = plt.figure(figsize=(10, 10), dpi=300)
ax = canvas.add_subplot(111)

def plot_erf(x: np.ndarray, *fx) -> None:
    STYLE = {
        'color':['#168039','#ff5252', '#ffbe43', '#a24cc2'],
        'linestyle':['-','--', '--', '--'],
        'legend':[
            'erf(x)', '$I_{1}(x)$', '$I_{2}(x)$', '$I_M(x)$'
        ],
        'xlabel':'$\it{X}$',
        'ylabel':'$\it{f(X)}$'
    }

    for i, f in enumerate(fx):
        ax.plot(x, f, color=STYLE['color'][i], linestyle=STYLE['linestyle'][i])

    plt.xlabel(STYLE['xlabel'])
    plt.ylabel(STYLE['ylabel'])
    plt.legend(STYLE['legend'], loc='best')
    plt.show()

def save_results(filename: str = "compared") -> None:
    canvas.savefig(filename+'.svg', format='svg')

def approximate_linear(x):
    erf_li = \
        x * ( math.exp(-x**2) + 1 ) / math.sqrt(math.pi)
    return erf_li

def approximate_square(x):
    erf_sqi = \
        x * ( math.exp(-x**2) + 1 + 4*math.exp(-(x/2)**2) ) / 3*math.sqrt(math.pi)
    return erf_sqi

def approximate_series(x):
    erf_series = \
        ( -(x**7)/42 + (x**5)/10 - (x**3)/3 + x) * (2 / math.sqrt(math.pi))
    return erf_series

def approximate(xlim, stepsize = 0.01, dtype = np.float32):
    x = np.arange(xlim, step=stepsize, dtype=dtype)
    
    erf_true =  np.array([math.erf(xi) for xi in x], dtype=dtype)
    erf_li =    np.array([approximate_linear(xi) for xi in x], dtype=dtype)
    erf_sqi =   np.array([approximate_square(xi) for xi in x], dtype=dtype)
    erf_series =  np.array([approximate_series(xi) for xi in x], dtype=dtype)

    plot_erf(x, erf_true, erf_li, erf_sqi, erf_series)

    # for factor in [2, 4, 5, 10, 20]:
    #     print(f'Рассчёт в диапазоне [0;{xlim}] Для x < {xlim/(factor)}')
    #     for couple in [(erf_true, erf_li), (erf_true, erf_sqi), (erf_true, erf_series)]:
    #         d = diff(*couple, len(x) // factor )
    #         print(f'Максимальное отклонение {d:e}')

def diff(f1, f2, xlim):
    return max([abs(f1i - f2i) for i, (f1i, f2i) in enumerate(zip(f1, f2)) if i < xlim])
