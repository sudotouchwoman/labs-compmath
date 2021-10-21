import numpy as np

class LinearMethod:
    X_NODES = None
    Y_NODES = None

    def __init__(self) -> None:
        pass

    def fit(self, x_nodes, y_nodes):
        if len(x_nodes) != len(x_nodes): raise ValueError
        if len(x_nodes) < 2: raise ValueError
        self.X_NODES = np.asarray(x_nodes)
        self.Y_NODES = np.asarray(y_nodes)
        self.A = x_nodes[0]
        self.B = x_nodes[-1]

    def predict_y(self, x_target):
        if self.X_NODES is None: raise AttributeError
        if self.Y_NODES is None: raise AttributeError
        if not self.A <= x_target <= self.B: raise ValueError

        def find_nearest(array: np.array, value: float) -> tuple:
            idx = np.searchsorted(array, value, side='left')
            if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
                return array[idx - 1], idx - 1
            else:
                return array[idx], idx
        
        x_i, i = find_nearest(self.X_NODES, x_target)
        x_nodes, y_nodes = self.X_NODES, self.Y_NODES

        k = (y_nodes[i+1] - y_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
        b = y_nodes[i]

        linear = lambda x: k * (x - x_i) + b

        return linear(x=x_target)

    def predict_ydx(self, x_target):
        if self.X_NODES is None: raise AttributeError
        if self.Y_NODES is None: raise AttributeError
        if not self.A <= x_target <= self.B: raise ValueError

        def find_nearest(array: np.array, value: float) -> tuple:
            idx = np.searchsorted(array, value, side='left')
            if idx > 0 and (idx == len(array) or np.fabs(value - array[idx-1]) < np.fabs(value - array[idx])):
                return array[idx - 1], idx - 1
            else:
                return array[idx], idx

        _, i = find_nearest(self.X_NODES, x_target)
        x_nodes, y_nodes = self.X_NODES, self.Y_NODES
        
        return (y_nodes[i+1] - y_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
        