from utils import brachistochrone

if __name__ == '__main__':
    approximator = brachistochrone.BrachistochroneErrorComputer('res/config/boundary-conds.json')
    approximator.set_model()
    results = approximator.compare_methods()
    approximator.plot_log_errors(*results)
    