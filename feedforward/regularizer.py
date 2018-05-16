from __future__ import division

import numpy as np

class Regularizer(object):
    def loss(self, W):
        raise NotImplementedError

    def backward(self, W):
        raise NotImplementedError


class L2(Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def loss(self, W):
        return np.sum(W * W) * self.strength

    def backward(self, W):
        return self.strength * 2 * W


class L1(Regularizer):
    def __init__(self, strength):
        self.strength = strength

    def loss(self, W):
        return np.sum(np.abs(W)) * self.strength

    def backward(self, W):
        out = np.ones(W.shape) * self.strength
        out[W < 0] *= -1
        return out
