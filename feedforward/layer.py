
import numpy as np

class Layer(object):
    def __init__(self, n_nodes):
        self.n_inputs = 0
        self.n_nodes = n_nodes
        self.W = np.zeros((1, 1))   # Weight matrix
        self.X = None               # Previous input
        self.A = None               # Previous activation
        self.weight_updater = None

    def init_weights(self, n_inputs, weight_updater):
        self.n_inputs = n_inputs
        self.weight_updater = weight_updater
        bound = 1/np.sqrt(n_inputs)  # Init with random weights, [-1 / sqrt(n), 1 / sqrt(n)]
        # self.W = np.ones((self.n_inputs, self.n_nodes))
        self.W = np.random.rand(self.n_inputs, self.n_nodes) * bound * 2 - bound

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dLdy):
        raise NotImplementedError


class FullSigmoidLayer(Layer):
    def __init__(self, n_nodes):
        super(FullSigmoidLayer, self).__init__(n_nodes)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDelta(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        # X = np.array([1, 2, 3]).reshape((1, 3))
        # self.W = np.array([1, 2, 3, 4, 5, 6]).reshape((3, 2))
        # X = np.array([1, 2, 3, 3, 2, 1]).reshape((2, 3))
        # self.W = np.array([1, 2, 3, 4, 5, 6]).reshape((3, 2))

        self.X = X
        self.A = X.dot(self.W)
        return self.sigmoid(self.A)

    def backward(self, dLdy):
        dLdh = (dLdy*self.sigmoidDelta(self.A)).dot(np.transpose(self.W))
        dLdW = np.transpose(self.X).dot(dLdy*self.sigmoidDelta(self.A))
        self.weight_updater.update(self.W, dLdW)
        return dLdh


class FullLinearLayer(Layer):
    def __init__(self, n_nodes):
        super(FullLinearLayer, self).__init__(n_nodes)

    def forward(self, X):
        self.X = X
        self.A = X.dot(self.W)
        return self.A

    def backward(self, dLdy):
        dLdh = (dLdy).dot(np.transpose(self.W))
        dLdW = np.transpose(self.X).dot(dLdy)
        self.weight_updater.update(self.W, dLdW)
        return dLdh
