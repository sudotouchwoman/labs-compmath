from scipy.optimize import root
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=5)
def find_constants(endpoint: tuple):
    xa, ya = endpoint

    def boundary_conds(interest):
        C, T = interest

        f = [
            C * (T - 0.5 * np.sin(2*T)) - xa,
            C * (0.5 - 0.5 * np.cos(2*T)) - ya
        ]
        return f

    solution = root(boundary_conds, [0.9, 1.0])
    return solution.x
    
def find_upper_bound(C:float, a:float):

        
    f = lambda t: 0.5 * C * (2*t - np.sin(2*t)) - a 
    solution = root(f, 0.9)
    
    return solution.x[0]
