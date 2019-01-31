import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import random

import network

def stochastic_gradient_descent(network, training_data, epochs, mini_batch_size, alpha, test_data = None):
    training_data = list(training_data)
    n = len(training_data)

    if test_data:
        test_data = list(test_data)
        n_test = len(test_data)

    for j in range(epochs):
        random.shuffle(training_data)
        
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
        
        for mini_batch in mini_batches:
            X = np.vstack([t[0] for t in mini_batch])
            Y = np.vstack([t[1] for t in mini_batch])

            network.forwardprop(X)
            network.backprop(Y, alpha)
        
        if test_data:
            print("Epoch {} : {} / {}".format(j, evaluate(network, test_data), n_test))
        else:
            print("Epoch {} complete".format(j))

def evaluate(network, test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""

    test_results = [(np.argmax(network.forwardprop(x)), y)
                    for (x, y) in test_data]

    return sum(int(x == y) for (x, y) in test_results)

if __name__ == '__main__':
    # setup the network and initialize with random weights
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net, W, B = network.network_random_gaussian([784, 30, 10])

    stochastic_gradient_descent(net, training_data, 30, 10, 3.0, test_data=test_data)

