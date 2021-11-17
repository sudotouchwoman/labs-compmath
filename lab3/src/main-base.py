import numpy as np
import matplotlib.pyplot as plt
from utils.odeint import solve_ode
import seaborn as sns
sns.set()

if __name__ == '__main__':
    thresh = 30
    modes = (
        (0.02, 0.2, -65, 6),
        (0.02, 0.25, -65, 6),
        (0.02, 0.2, -50, 2),
        (0.1, 0.2, -65, 2)
    )

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20,20))
    axes[0,0].set_title(r'Tonic spiking')
    axes[0,1].set_title(r'Phasing spiking')
    axes[1,0].set_title(r'Chattering')
    axes[1,1].set_title(r'Fast spiking')

    for i, mode in enumerate(modes):
        a, b, c, d = mode

        def constraint(t, X):
            v, u = X
            if v > thresh:
                v, u = c, u + d
            return np.asarray([v, u])

        def f(t, X, I=5):
            v, u = X
            dvdt = 0.04*(v*v) + 5*v + 140 - u + I
            dudt = a*(b*v - u)
            return np.asarray([dvdt, dudt])
        
        results = solve_ode([c, b*c], 300, f, constraint=constraint, h=.1, method='euler')
        axes[divmod(i, 2)].plot(
            results['t'],
            results['y'][:,0],
            label=r'Forward Euler',
            marker='o',
            linestyle=':',
            color='#F24162',
            alpha=.5)
        
        results = solve_ode([c, b*c], 300, f, constraint=constraint, h=.1, method='imp-euler')
        axes[divmod(i, 2)].plot(
            results['t'],
            results['y'][:,0],
            label=r'Implicit (backward) Euler',
            marker='o',
            linestyle=':',
            color='#58F380',
            alpha=.5)

        results = solve_ode([c, b*c], 300, f, constraint=constraint, h=.1, method='runge-kutta')
        axes[divmod(i, 2)].plot(
            results['t'],
            results['y'][:,0],
            label=r'Runge-Kutta',
            marker='o',
            linestyle=':',
            color='#092E51',
            alpha=.5)

    for ax in axes:
        ax[0].set_ylim([-100, 50])
        ax[1].set_ylim([-100, 50])
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')

    fig.tight_layout()

    fig.savefig(f'res/img/neuron-modes-constrainted.svg', format='svg')
