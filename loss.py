from __future__ import division

import numpy as np
from layer import Layer

class Loss(Layer):
    def __init__(self, n_outputs):
        super(Loss, self).__init__(n_outputs)
        self.activation = None
        self.input = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDelta(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def loss(self, X, D):
        self.input = X
        self.activation = X.dot(self.W)
        y = self.sigmoid(self.activation)

        N = X.shape[0]
        error = 1/N * np.transpose(y - D).dot(y - D)
        return error

    def backward(self, X, D):
        X = np.ones((1, 3))

        a = X.dot(self.W)
        y = a #self.sigmoid(a)

        N = X.shape[0]
        dLdy = 1/N * np.ones((1, N)).dot(2 * (D - y))

        # TODO must stack deriv N times before multiplication
        deriv = self.sigmoidDelta(self.activation)
        dydw = np.transpose(self.input).dot(derivXN)
        print(dLdy)


