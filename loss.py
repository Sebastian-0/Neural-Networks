import numpy as np
from layer import Layer

class Loss(Layer):
    def __init__(self, n_outputs):
        super(Loss, self).__init__(n_outputs)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, X, D):
        a = np.transpose(self.W).dot(X)
        y = self.sigmoid(a)

        N = X.shape[0]
        error = 1/N * np.transpose(y - D).dot(y - D)
        return error
