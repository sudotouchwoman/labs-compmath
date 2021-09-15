def L_at_x(x:float, x_nodes, y_nodes) -> float:
    '''
        Return `L(x)` value for Lagrange polynom on given set of nodes at given x

        + `x`: float
            value to compute for

        + `x_nodes`: list or np.ndarray
            set of x-axis values of interpolation nodes
            
        + `y_nodes`: list or np.ndarray
            set of y-axis values of interpolation nodes
    '''
    def l_i_at_x(i:int, x:float, x_nodes) -> float:
        # Compute i-th Lagrange base polynom value on provided set of x-axis for given x
        # `i`: int
        #   index of polynom
        # `x`: float
        #   value to compute for
        # `x_nodes`: list or np.ndarray
        #   set of x-axis values of interpolation nodes
        polynom_value = 1.0
        for j, node_x in enumerate(x_nodes):
            if j == i: continue
            # I considered iterative multiplication to be more stable and easy-to-implement
            # however, the overall method is noticably slow for repeated computations as
            # we have O(nodes^2) in total, multiplied by amount of x to compute for
            polynom_value *= ((x - node_x)/(x_nodes[i] - node_x))
        return polynom_value

    # at last, perform summation of base polynom values
    return sum([( y_nodes[i] * l_i_at_x(i, x, x_nodes) ) for i, _ in enumerate(x_nodes)])