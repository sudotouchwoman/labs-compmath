import numpy as np

def derivative1(x_nodes, y_nodes) -> np.array:
    # compute first derivative for given collections of nodes
    # assume that the absciasses are distributed uniformly
    # compute using the formulae of 2nd accuracy degree (requires at least 3 nodes)
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    h = x_nodes[1] - x_nodes[0]
    nodes = len(x_nodes)
    if nodes < 3: raise ValueError('Usage: at least three nodes')
    if len(x_nodes) != len(y_nodes): raise ValueError('x_nodes and y_nodes must be of same shape')

    def d1_generator():
        dx0 = lambda: (-3 * y_nodes[0] + 4 * y_nodes[1] - y_nodes[2]) / (2 * h)
        dxn = lambda : (y_nodes[-3] - 2 * y_nodes[-2] + 3*y_nodes[-1]) / (2 * h)
        dx = lambda i: (y_nodes[i+1] - y_nodes[i-1]) / (2 * h)

        yield dx0()
        for i in range(1, nodes - 1):
            yield dx(i)
        yield dxn()

    d1_nodes = np.array(list(d1_generator()))
    return d1_nodes