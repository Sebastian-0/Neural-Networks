from __future__ import division

import numpy as np

class Loss(object):
    def loss(self, Y, D):
        raise NotImplementedError

    def backward(self, Y, D):
        raise NotImplementedError


class MeanSquareLoss(Loss):
    def loss(self, Y, D):
        N = Y.shape[0]
        delta = D - Y
        error = 1 / N * np.sum(delta * delta)
        return error

    def backward(self, Y, D):
        # Y = np.array([22, 28]).reshape((1, 2))
        # D = np.array([5, 10]).reshape((1, 2))
        # Y = np.array([22, 28, 14, 20]).reshape((2, 2))
        # D = np.array([5, 10, 10, 5]).reshape((2, 2))

        P = Y.shape[0]
        dLdy = -2 / P * (D - Y)
        return dLdy



