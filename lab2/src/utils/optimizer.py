from scipy.optimize import root
import numpy as np

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
