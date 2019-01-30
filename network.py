import numpy as np

class Network:
    # number of layers
    num_layers = 1

    # the weight matrices
    weights = []

    # list of vectors holding the state
    state = []

    def __init__(self, layers):
        """Create a network from an array of layer sizes"""
        self.num_layers = len(layers)

        for i in range(self.num_layers - 1):
            # We initialise the weights with 0
            # a list of the weights of the layers
            # size num-layers - 1 as each weights matrix govers the transition between layers
            # each item in W is a matrix with dimensions len(layer+1) * (len(layer)+1)
            self.weights.append(np.zeros([layers[i + 1], layers[i] + 1]))

            # the state of each layer
            self.state.append(np.zeros(layers[i]))

        # append one more state layer for the output
        self.state.append(np.zeros(layers[self.num_layers - 1]))

    def input(self, inputs):
        self.state[0] = inputs

    def forwardprop(self):
        """vectorized implementation of forward propagation"""
        # for each layer
        for i in range(self.num_layers - 1):
            # pad the state with a 1 at the start
            X = pad_ones(self.state[i])

            # multiply the weights by the input
            Z = np.matmul(X, np.transpose(self.weights[i]))

            # apply the sigmoid
            self.state[i + 1] = sigmoid(Z)
        
        return self.state[self.num_layers - 1]


def sigmoid(z):
    """vectorized implementation of the sigmoid function"""
    return 1 / (1 + np.exp(-z))

def pad_ones(data):
    """adds an entry / column of 1s for the biases"""
    if data.ndim > 1:
        return np.insert(data, 0, 1, axis=1)
    else:
        return np.insert(data, 0, 1)


if __name__ == '__main__':
    # If this was run as a main script
    # Test the network running xor
    net_xor = Network([2, 2, 1])
    net_xor.weights[0] = np.array([[-30, 20, 20], [10, -20, -20]])
    net_xor.weights[1] = np.array([[-10, 20, 20]])

    net_xor.input(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    print(net_xor.forwardprop())