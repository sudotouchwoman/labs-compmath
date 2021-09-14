from utils import interpolate

if __name__ == '__main__':
    NODES = interpolate.load_nodes('res/nodes.json')
    SplineModel = interpolate.CubSqplineMethod(*NODES)
    SplineModel.plot_results('res/spline')