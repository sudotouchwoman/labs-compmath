from utils import error

if __name__ == '__main__':
    approximator = error.BrachistochroneErrorComputer('res/config/boundary-conds.json')
    approximator.set_model()
    results = approximator.compare_methods()
    approximator.plot_errors_logscale(*results)
    