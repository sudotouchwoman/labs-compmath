from utils import interpolate

if __name__ == '__main__':
    NODES = interpolate.load_nodes('res/cfg/nodes.json')
    SplineModel = interpolate.CubSqplineMethod(*NODES)
    SplineModel.interpolate('res/plots/spline')

    LagrangeModel = interpolate.LagrangeMethod(*NODES)
    LagrangeModel.interpolate('res/plots/lagrange')