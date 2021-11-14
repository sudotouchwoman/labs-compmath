from utils.neurons import NeuralNetwork, draw_firings

if __name__ == '__main__':
    network = NeuralNetwork(excitatory_ns=40, total_ns=100)
    firings = network.simulate(t_n=100., h=0.5)
    draw_firings(firings)