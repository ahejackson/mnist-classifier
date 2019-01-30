# OUTLINE

1. Load MNIST data
2. Setup network - initialise with random weights and biases
3. Train network using stochastic batch gradient descent

3.1 Randomly select a subset of elements from the full training set
3.2 Train against these elements

4. Repeat until...?
5. Test against the test set




# THE DATA

The MNIST training set:
http://yann.lecun.com/exdb/mnist/

* handwritten digits collected by NIST (National Institute of Standards and Technology) in the US
* 60,000 example training set and 10,000 example test set
* 28x28 greyscale pixels


# NETWORK ARCHITECTURE

Following the example of the Neural Networks and Deep Learning book
http://neuralnetworksanddeeplearning.com/chap1.html

This is a three layer network:
* 784 neurons in the input layer corresponding to the 784 pixels on a 28x28 pixel input image, each one taking in a greyscale intensity from 0 to 1
* 15 neurons in the hidden layer
* 10 neurons in the output layer corresponding to logistic regression estimates for 0 to 10

The network will guess the number corresponding to the output it expresses the most confidence in