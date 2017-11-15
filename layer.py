
import numpy as np

class Layer(object):
    def __init__(self, n_nodes):
        self.n_inputs = 0
        self.n_nodes = n_nodes
        self.W = np.zeros((1, 1))

    def init_weights(self, n_inputs):
        self.n_inputs = n_inputs
        bound = 1/np.sqrt(n_inputs)  # Init with random weights, [-1 / sqrt(n), 1 / sqrt(n)]
        self.W = np.ones((self.n_inputs, self.n_nodes)) #np.random.rand(self.n_inputs, self.n_nodes) * bound * 2 - bound

    def forward(self, X):
        raise NotImplementedError

    def backward(self, D):
        raise NotImplementedError


class FullSigmoidLayer(Layer):
    def __init__(self, n_nodes):
        super(FullSigmoidLayer, self).__init__(n_nodes)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        a = X.dot(self.W)
        return self.sigmoid(a)
