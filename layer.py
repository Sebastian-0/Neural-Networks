
import numpy as np

class Layer(object):
    def __init__(self, n_nodes):
        self.n_inputs = 0
        self.n_nodes = n_nodes
        self.W = np.zeros((1, 1))   # Weight matrix
        self.X = None               # Previous input
        self.A = None               # Previous activation

    def init_weights(self, n_inputs):
        self.n_inputs = n_inputs
        bound = 1/np.sqrt(n_inputs)  # Init with random weights, [-1 / sqrt(n), 1 / sqrt(n)]
        # self.W = np.ones((self.n_inputs, self.n_nodes)) #np.random.rand(self.n_inputs, self.n_nodes) * bound * 2 - bound
        self.W = np.random.rand(self.n_inputs, self.n_nodes) * bound * 2 - bound

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dLdy):
        raise NotImplementedError

    def update_weights(self, delta):  # delta is the derivative of loss w. respect to W
        pass


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
        return dLdh

        # dydx = np.transpose(np.tile(self.activation, (X.shape[1], 1)))
        # dLdX = dLdy.dot(dydx)
        #
        # delta = self.activation  # self.sigmoidDelta(self.activation)
        # print(delta)
        # dydw = np.transpose(self.input).dot(delta)
        #
        # self.update_weights(dLdy * dydw)
        # print(dLdy * dydw)
