import numpy as np
import matplotlib.pyplot as plt

class Network:
    # %timeit time_network()
    # 151 ms ± 291 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    # number of layers
    num_layers = 1

    # the size of each layer
    layer_size = []

    # the weight matrices
    weights = []

    # list of vectors holding the activations
    activations = []
    weighted_inputs = []
    error = []

    def __init__(self, layers):
        """Create a network from an array of layer sizes"""
        self.num_layers = len(layers)
        self.layer_size = np.array(layers, copy=True)

        for i in range(self.num_layers - 1):
            # We initialise the weights with 0
            # a list of the weights of the layers
            # size num-layers - 1 as each weights matrix govers the transition between layers
            # each item in W is a matrix with dimensions len(layer+1) * (len(layer)+1)
            self.weights.append(np.zeros([layers[i + 1], layers[i] + 1]))

            # the activations of each layer
            self.activations.append(np.zeros(layers[i]))
            self.weighted_inputs.append(np.zeros(layers[i]))
            self.error.append(np.zeros(layers[i]))

        # append one more activations layer for the output
        self.activations.append(np.zeros(layers[self.num_layers - 1]))
        self.weighted_inputs.append(np.zeros(layers[self.num_layers - 1]))
        self.error.append(np.zeros(layers[self.num_layers - 1]))

    def input(self, inputs):
        self.activations[0] = inputs

    def forwardprop(self):
        """vectorized implementation of forward propagation"""
        # for each layer
        for i in range(self.num_layers - 1):
            # pad the activations with a 1 at the start
            A = pad_bias(self.activations[i])

            # multiply the weights by the input
            self.weighted_inputs[i + 1] = np.matmul(A, np.transpose(self.weights[i]))

            # apply the sigmoid
            self.activations[i + 1] = sigmoid(self.weighted_inputs[i + 1])
        
        return self.activations[self.num_layers - 1]

    def output(self):
        """returns the output of the network (the final layer of activations)"""
        return self.activations[self.num_layers - 1]

    def cost(self, X, Y):
        """returns the cost function for each matched pair of training inputs X and outputs Y"""
        # the number of training examples will be Y.shape[0] (or X.shape[0] - they should be equal)
        self.input(X)
        self.forwardprop()
        return np.sum((Y - self.output()) ** 2) / (2 * Y.shape[0])

    def backprop(self, Y, alpha):
        """vectorized implementation of backwards propagation"""
        # index of the last layer
        L = self.num_layers - 1
        
        # size of the example set
        m = Y.shape[0]

        # 1. Calculate the error of the output layer
        # Returns a 2 dimensional array of errors
        # Each row corresponds to the errors of the row in the training sample
        # The elements of each row vector are the errors of each neuron on the output layer
        self.error[L] = self.activations[L] * sigmoid_prime(self.weighted_inputs[L])

        # backpropagate the error
        # note this loop terminates before 0 - 1 is the lowest value l will take
        for l in range(L - 1, 0, -1):
            # find the error in layer l in terms of the error in layer l+1
            # have to replace the weight matrix with a version with the bias column deleted
            # next iteration of this - separate the bias and the weights
            self.error[l] = np.matmul(self.error[l + 1], unpad_bias(self.weights[l])) * sigmoid_prime(self.weighted_inputs[l])
        
        # loop back through the layers again, updating the weights
        for l in range(L, 0, -1):
            # pad the activations for the layer with ones again
            A = pad_bias(self.activations[l - 1])

            # calculate the change in the weights for this layer
            dW = np.matmul(np.transpose(self.error[l]), A)

            # update with gradient descent
            self.weights[l - 1] -= (alpha / m) * dW

# UTILITY FUNCTIONS

def sigmoid(z):
    """vectorized implementation of the sigmoid function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """vectorized implementation of the derivative of the sigmoid function"""
    return np.exp(-z) / (1 + np.exp(-z))**2

def pad_bias(data):
    """adds an entry / column of 1s for the biases"""
    if data.ndim > 1:
        return np.insert(data, 0, 1, axis = 1)
    else:
        return np.insert(data, 0, 1)

def unpad_bias(data):
    """removes the bias column"""
    return np.delete(data, 0, axis = 1)

# NETWORK OPERATION FUNCTIONS

def simple_test():
    # If this was run as a main script
    # Test the network running xnor
    n = Network([2, 2, 2])
    n.weights[0] = np.array([[-30., 20., 20.], [10., -20., -20.]])
    n.weights[1] = np.array([[-10., 20., 20.], [10., -20., -20.]])

    # The input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    n.input(X)

    cost = n.cost(X, Y)
    n.backprop(Y, 0.1)
    print("cost ",cost)

def setup_test_network():
    n = Network([2, 2, 2])
    W0 = (np.random.rand(2, 3) * 50) - 25
    W1 = (np.random.rand(2, 3) * 50) - 25
    n.weights[0] = W0
    n.weights[1] = W1

    return n, W0, W1

def iterate_network(n, X, Y, alpha = 0.1, iterations = 1000):
    J = []

    for i in range(0, iterations):
        J.append(n.cost(X, Y))
        n.backprop(Y, alpha)
    return J

def plot_cost(J):
    # plot the results
    x = np.arange(0, len(J))

    plt.plot(x, J, label='cost')

    plt.xlabel('iterations')
    plt.ylabel('cost')

    plt.title("Cost function over time")
    plt.legend()
    plt.show()

def time_network():
    # The input and output
    n, W0, W1 = setup_test_network()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    J = iterate_network(n, X, Y, 0.1, 1000)

if __name__ == '__main__':
    # The input and output
    n, W0, W1 = setup_test_network()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    J = iterate_network(n, X, Y, 0.1, 1000)
    plot_cost(J)