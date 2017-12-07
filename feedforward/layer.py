
import numpy as np

class Layer(object):
    def __init__(self, n_nodes):
        self.n_inputs = 0
        self.n_nodes = n_nodes
        self.W = np.zeros((1, 1))   # Weight matrix
        self.b = np.zeros((1, 1))   # Bias weights
        self.X = None               # Previous input
        self.A = None               # Previous activation
        self.weight_updater = None

    def init_weights(self, n_inputs, weight_updater):
        self.n_inputs = n_inputs
        self.weight_updater = weight_updater
        bound = 1/np.sqrt(n_inputs)  # Init with random weights, [-1 / sqrt(n), 1 / sqrt(n)]
        # self.W = np.ones((self.n_inputs, self.n_nodes))
        self.W = np.random.rand(self.n_inputs, self.n_nodes) * bound * 2 - bound
        self.b = np.random.rand(1, self.n_nodes) * bound * 2 - bound

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dLdy):
        raise NotImplementedError


class FullLayer(Layer):
    def __init__(self, n_nodes):
        super(FullLayer, self).__init__(n_nodes)

    def activation(self, x):
        raise NotImplementedError

    def activationDelta(self, x):
        raise NotImplementedError

    def forward(self, X):
        self.X = X
        self.A = X.dot(self.W) + self.b
        return self.activation(self.A)

    def backward(self, dLdy):
        dLdh = (dLdy*self.activationDelta(self.A)).dot(np.transpose(self.W))
        dLdW = np.transpose(self.X).dot(dLdy*self.activationDelta(self.A))
        dLdb = np.sum(dLdy*self.activationDelta(self.A), axis=0)  # TODO was here with bias
        self.weight_updater.update(self.W, self.b, dLdW, dLdb)
        return dLdh


class FullSigmoidLayer(FullLayer):
    def __init__(self, n_nodes):
        super(FullSigmoidLayer, self).__init__(n_nodes)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activationDelta(self, x):
        s = self.activation(x)
        return s * (1 - s)


class FullLinearLayer(FullLayer):
    def __init__(self, n_nodes):
        super(FullLinearLayer, self).__init__(n_nodes)

    def activation(self, x):
        return x

    def activationDelta(self, x):
        return np.ones(x.shape)
