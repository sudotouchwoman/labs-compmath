import matplotlib.pyplot as plt
import numpy as np
import math

canvas = plt.figure(figsize=(10, 10), dpi=300)
ax = canvas.add_subplot(111)

def plot_erf(x: np.ndarray, dtype = np.float32, *fx) -> None:
    STYLE = {
        'color':['#168039','#ff5252', '#ffbe43', '#A24CC2'.lower()],
        'linestyle':['-','--', '--', '--'],
        'legend':[
            'erf(x)', 'lin approx', 'squared approx', 'approx from wiki'
        ]
    }
    for i, f in enumerate(fx):
        ax.plot(x, f, color=STYLE['color'][i], linestyle=STYLE['linestyle'][i])

    plt.xlabel('$\it{X}$')
    plt.ylabel('$\it{erf(X)}$')
    plt.legend(STYLE['legend'], loc='best')
    plt.show()

def plot_error(x: np.ndarray, dtype = np.float32, *fx) -> None:
    STYLE = {
        'legend':[
            '|erf(x) - lin approx|', '|erf(x) - squared approx|', '|erf(x) - approx from wiki|'
        ]
    }
    reference = fx[0]
    for i, f in enumerate(fx[1:]):
        ax.plot(x, abs(f - reference), linestyle=STYLE['linestyle'][i])

    plt.xlabel('$\it{X}$')
    plt.ylabel('Error |$\it{erf(X) - erf*(X)}$|')
    plt.legend(STYLE['legend'], loc='best')
    plt.show()

def save_results(filename: str = "compared") -> None:
    canvas.savefig(filename+'.svg', format='svg')

def approximate_linear(x, dtype = np.float32):
    erf_li = \
        x * ( math.exp(-x**2) + 1 ) / math.sqrt(math.pi)
    return erf_li

def approximate_square(x, dtype = np.float32):
    erf_sqi = \
        x * ( math.exp(-x**2) + 1 + math.exp(-(x/2)**2) ) / 3*math.sqrt(math.pi)
        #( 2.0 / math.sqrt(math.pi) ) * ( x + ( math.exp((-x**2) / 4) - 1 )*(2*math.exp(-x**2) - 2*math.exp((-x**2) / 4) + x) / 3 )
    return erf_sqi

def approximate_wiki(x, dtype = np.float32):
    a = \
        (8 / 3 * math.pi)*((3 - math.pi)/(math.pi - 4))
    erf_sqi = \
        math.sqrt( 1 - math.exp( (-x**2)*(4/math.pi + a*x**2)/(1 + a*x**2) ) )
        #( 2.0 / math.sqrt(math.pi) ) * ( x + ( math.exp((-x**2) / 4) - 1 )*(2*math.exp(-x**2) - 2*math.exp((-x**2) / 4) + x) / 3 )
        #x * ( math.exp(-x**2) + math.exp((-x**2) / 4) + 1 ) / 3*math.sqrt(math.pi)
    return erf_sqi

def approximate(xlim, stepsize = 0.01, dtype = np.float32):
    x = np.arange(xlim, step=stepsize, dtype=dtype)
    
    erf_true =  np.array([math.erf(xi) for xi in x])
    erf_li =    np.array([approximate_linear(xi, dtype) for xi in x])
    erf_sqi =   np.array([approximate_square(xi, dtype) for xi in x])
    erf_wiki =  np.array([approximate_wiki(xi, dtype) for xi in x])

    plot_erf(x, dtype, erf_true, erf_li, erf_sqi, erf_wiki)
    #plot_error(x, dtype, erf_true, erf_li, erf_sqi, erf_wiki)