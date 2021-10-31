from utils.discrete import DiscreteOptimizer

if __name__ == '__main__':

    optimizer = DiscreteOptimizer('res/config/discrete-conds.json')
    surface_properties = optimizer.create_surfaces()
    optimizer.draw_surfaces(*surface_properties)

    # uncomment below to get plots of comparison 
    # between true curve and the linear approximation
    # and their derivatives
    # optimizer.peek_plots()
