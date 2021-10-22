from utils.discrete import DiscreteOptimizer

if __name__ == '__main__':

    optimizer = DiscreteOptimizer('res/config/discrete-conds.json')
    optimizer.peek_plots()