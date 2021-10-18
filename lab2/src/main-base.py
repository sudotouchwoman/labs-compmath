from utils import brachistochrone

if __name__ == '__main__':
    approximator = brachistochrone.BrachistochroneApproximator('res/cfg/boundary-conds.json')
    approximator.set_model()
    # import numpy as np
    # integral = \
    #     np.sqrt(1.034399843373137) * (1.034399843373137) * (2*1.754184384262122 - np.sin(2*1.754184384262122))
    # print(integral)