from scipy.integrate import quad

from .quadratures.simpson import composite_simpson
from .quadratures.trapezoid import composite_trapezoid

def compare():
    f = lambda x: 5*x**2 - 3*x
    integral = composite_simpson(0, 1, 1000, f)
    print(f'Simpson result: {integral}')
    integral = composite_trapezoid(0, 1, 1000, f)
    print(f'Trapezoid result: {integral}')
    integral = quad(f, 0, 1)
    print(f'Area is {integral[0]}')
