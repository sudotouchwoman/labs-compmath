from utils import brachistochrone

if __name__ == '__main__':
    approximator = brachistochrone.BrachistochroneApproximator('res/config/boundary-conds.json')
    approximator.set_model()
    approximator.compare_methods()
    