import numpy as np
import matplotlib.pyplot as plt

class Network:
    # %timeit test_network_xor()
    # 162 ms ± 3.78 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # number of layers
    num_layers = 1

    def __init__(self, layers):
        """Create a network from an array of layer sizes"""
        self.num_layers = len(layers)
        self.sizes = np.array(layers, copy=True)

        # the weight matrices
        self.weights = []
        self.biases = []

        # list of vectors holding the activations
        self.activations = []
        self.weighted_inputs = []
        self.error = []

        for i in range(self.num_layers - 1):
            # We initialise the weights with 0
            # a list of the weights of the layers
            # size num-layers - 1 as each weights matrix govers the transition between layers
            # each weight matrix has dimension len(layer i+1) * (len(layer i))
            # each bias is a len(layer i + 1) x 1 column vector 
            self.weights.append(np.zeros([layers[i + 1], layers[i]]))
            self.biases.append(np.zeros([layers[i + 1], 1]))

            # the activations of each layer
            self.activations.append(np.zeros(layers[i]))
            self.weighted_inputs.append(np.zeros(layers[i]))
            self.error.append(np.zeros(layers[i]))

        # append one more activations layer for the output
        self.activations.append(np.zeros(layers[self.num_layers - 1]))
        self.weighted_inputs.append(np.zeros(layers[self.num_layers - 1]))
        self.error.append(np.zeros(layers[self.num_layers - 1]))

    def input(self, inputs):
        """Provide input to the network"""
        self.activations[0] = inputs

    def forwardprop(self, X):
        """vectorized implementation of forward propagation"""
        self.activations[0] = X
        
        # for each layer
        for i in range(self.num_layers - 1):
           # multiply the weights by the input
            self.weighted_inputs[i + 1] = np.matmul(self.activations[i], np.transpose(self.weights[i])) + np.transpose(self.biases[i])

            # apply the sigmoid
            self.activations[i + 1] = sigmoid(self.weighted_inputs[i + 1])
        
        return self.activations[self.num_layers - 1]

    def output(self):
        """returns the output of the network (the final layer of activations)"""
        return self.activations[self.num_layers - 1]

    def cost(self, X, Y):
        """returns the cost function for each matched pair of training inputs X and outputs Y"""
        # the number of training examples will be Y.shape[0] (or X.shape[0] - they should be equal)
        self.forwardprop(X)
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
        self.error[L] = (self.activations[L] - Y) * sigmoid_prime(self.weighted_inputs[L])

        # backpropagate the error
        # note this loop terminates before 0 - 1 is the lowest value l will take
        for l in range(L - 1, 0, -1):
            # find the error in layer l in terms of the error in layer l+1
            # have to replace the weight matrix with a version with the bias column deleted
            # next iteration of this - separate the bias and the weights
            self.error[l] = np.matmul(self.error[l + 1], self.weights[l]) * sigmoid_prime(self.weighted_inputs[l])
        
        # loop back through the layers again, updating the weights
        for l in range(L, 0, -1):
            # update with gradient descent
            self.weights[l - 1] -= (alpha / m) * np.matmul(np.transpose(self.error[l]), self.activations[l - 1])
            
            # update biases with gradient descent
            self.biases[l - 1] -= (alpha / m) * np.matmul(np.transpose(self.error[l]), np.ones([m, 1]))
    
    def iterate_network(self, X, Y, alpha = 0.1, iterations = 1000):
        """Iterates the network with the given learning rate alpha for the given iterations"""
        J = []

        for i in range(0, iterations):
            J.append(self.cost(X, Y))
            self.backprop(Y, alpha)
        return J

# CALCULATION FUNCTIONS
def sigmoid(z):
    """vectorized implementation of the sigmoid function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """vectorized implementation of the derivative of the sigmoid function"""
    return np.exp(-z) / (1 + np.exp(-z))**2

# NETWORK CREATION FUNCTIONS
def setup_xor_network():
    """Setup a network that calculates XNOR and XOR"""
    n = Network([2, 2, 2])
    n.weights[0] = np.array([[20., 20.], [-20., -20.]])
    n.biases[0] = np.array([[-30.], [10.]])

    n.weights[1] = np.array([[20., 20.], [-20., -20.]])
    n.biases[1] = np.array([[-10.], [10.]])
    return n

def xor_data():
    # The input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    return X, Y

def network_random_gaussian(layers):
    """Setup a network with all weights randomly chosen from gaussian distribution"""
    n = Network(layers)

    for i in range(n.num_layers - 1):
        n.weights[i] = np.random.randn(layers[i + 1], layers[i])
        n.biases[i] = np.random.randn(layers[i + 1], 1)

    return n, np.array(n.weights, copy=True), np.array(n.biases, copy=True)

# NETWORK TESTING FUNCTIONS
def plot_cost(J):
    """Plot the cost function against iterations"""
    x = np.arange(0, len(J))

    plt.plot(x, J, label='cost')

    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.ylim(0,1)

    plt.title("Cost function over time")
    plt.legend()
    plt.show()

def test_network_xor(alpha = 0.1, iterations = 1000):
    """Creates and trains a network against the XOR/XNOR data"""
    n, W, B = network_random_gaussian([2, 2, 2])
    X, Y = xor_data()

    return n.iterate_network(X, Y, alpha, iterations)

def quicktest(alpha = 0.1, iterations = 1000):
    """Convenience function to test and plot an XOR/XNOR network""" 
    plot_cost(test_network_xor(alpha, iterations))

if __name__ == '__main__':
    # The input and output
    n, W, B = network_random_gaussian([2, 2, 2])
    X, Y = xor_data()

    J = n.iterate_network(X, Y, 0.1, 1000)
    plot_cost(J)