'''
Perform simple accuracy test: integrate quadratures for exp(x)
The results show that implemented methods have corresponding tangents
It was also observed that if the function given is not quite smooth!
For polynom of degree 2 simpson error scaled to 1e-13 almost instantly
'''
import numpy as np

if __name__ == '__main__':
    from simpson import composite_simpson
    from trapezoid import composite_trapezoid
    from plotting import PlotArtist

    y = lambda x: np.exp(x)
    int_y = lambda x: np.exp(x)
    a, b = 0.0, 5.0

    error = lambda a, b, e: np.abs(e - (int_y(b) - int_y(a)))
    step = lambda a, b, n: (b - a) / (n - 1)

    n_values = np.arange(3, 2000, 10)
    steps = np.zeros_like(n_values, dtype=float)
    simpson_results = np.zeros_like(n_values, dtype=float)
    trap_results = np.zeros_like(n_values, dtype=float)

    for i, n in enumerate(n_values):
        
        errored = composite_simpson(a, b, n, y)
        simpson_results[i] = error(a, b, errored)
        
        errored = composite_trapezoid(a, b, n, y)
        trap_results[i] = error(a, b, errored)

        steps[i] = step(a, b, n)

    artist = PlotArtist()

    artist.add_log_plot(steps, trap_results, style={
        'legend':'Trapezoid error',
        'color':'#67CC8E',
        'linestyle':'-'
        })
    artist.add_log_plot(steps, simpson_results, style={
        'legend':'Simpson error',
        'color':'#9250BC',
        'linestyle':'-'
        })

    artist.save_as('quad-test', format='svg')
