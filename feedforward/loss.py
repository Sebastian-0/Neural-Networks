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
        error = 1/P * np.sum(delta * delta)
        return error

    def backward(self, Y, D):
        P = Y.shape[0]
        dLdy = -2/P * (D - Y)
        return dLdy


class CrossEntropyLoss(Loss):
    def loss(self, Y, D):
        P, M = Y.shape
        if M == 1:
            error = -1/P * np.sum(D*np.log(Y) + (1 - D)*np.log(1 - Y))
            return error
        else:
            raise NotImplementedError

    def backward(self, Y, D):
        P, M = Y.shape
        if M == 1:
            dLdy = 1/P * (Y - D) / (Y * (1 - Y))
            return dLdy
        else:
            raise NotImplementedError
