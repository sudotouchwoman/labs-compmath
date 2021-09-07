from utils import approx

if __name__ == '__main__':
    approx.approximate(xlim=2, stepsize=0.001)
    approx.save_results('results')