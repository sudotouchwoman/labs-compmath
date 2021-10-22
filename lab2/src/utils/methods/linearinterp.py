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
        return self

    def predict_y(self, x_target):
        if self.X_NODES is None: raise AttributeError
        if self.Y_NODES is None: raise AttributeError
        if not self.A <= x_target <= self.B: raise ValueError

        def find_nearest(array: np.array, value: float) -> tuple:
            idx = np.searchsorted(array, value, side='right')
            if idx == len(array): idx -= 1
            return array[idx - 1], idx - 1
        
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
            idx = np.searchsorted(array, value, side='right')
            if idx == len(array): idx -= 1
            return array[idx - 1], idx - 1

        _, i = find_nearest(self.X_NODES, x_target)
        x_nodes, y_nodes = self.X_NODES, self.Y_NODES
        
        return (y_nodes[i+1] - y_nodes[i]) / (x_nodes[i+1] - x_nodes[i])
        
if __name__ == '__main__':
    from plotting import PlotArtist

    model = LinearMethod()

    a, b = 0.0, 4.0
    y = lambda x: x**2 - 5*x + 3
    y = np.vectorize(y)
    h = 1e-2

    x_nodes = np.linspace(a, b, 4)
    y_nodes = y(x_nodes)
    y_test, x_test = [], []
    for x in np.arange(a, b + h, step=h):
        predicted = model.fit(x_nodes, y_nodes).predict_y(x)
        x_test.append(x)
        y_test.append(predicted)

    artist = PlotArtist()
    
    x_nodes = np.arange(a, b + h, step=h)
    y_nodes = y(x_nodes)

    artist.add_plot(x_test, y_test, style={
        'legend':'Model prediction',
        'color':'#343642',
        'linestyle':'--'
        })
    artist.add_plot(x_nodes, y_nodes, style={
        'legend':'Actual function',
        'color':'#962D3E',
        'linestyle':'-'
        })

    artist.save_as('linear-test', format='svg')

