from utils import lagrange

if __name__ == '__main__':
    lagrange.interpolate(stepsize=0.001)
    lagrange.save_results('results')