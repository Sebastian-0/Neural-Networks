from __future__ import division

import numpy as np

class Loss(object):
    def loss(self, Y, D):
        raise NotImplementedError

    def backward(self, Y, D):
        raise NotImplementedError


class MeanSquareLoss(Loss):
    def loss(self, Y, D):
        P = Y.shape[0]
        delta = D - Y
        error = 1 / P * np.sum(delta * delta)
        return error

    def backward(self, Y, D):
        P = Y.shape[0]
        dLdy = -2 / P * (D - Y)
        return dLdy
