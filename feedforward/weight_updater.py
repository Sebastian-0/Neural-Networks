
import numpy as np

class WeightUpdater(object):
    def update(self, W, dLdW, dLdb):
        raise NotImplementedError


class SGDUpdater(WeightUpdater):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, W, b, dLdW, dLdb):
        W -= self.learning_rate * dLdW
        b -= self.learning_rate * dLdb


class MomentumUpdater(WeightUpdater):
    def __init__(self, learning_rate=0.01, alpha=0.5):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.previous_W = None
        self.previous_b = None

    def update(self, W, b, dLdW, dLdb):
        delta_W = -self.learning_rate * dLdW
        delta_b = -self.learning_rate * dLdb
        if not self.previous_W is None:
            delta_W += self.alpha * self.previous_W
        if not self.previous_b is None:
            delta_b += self.alpha * self.previous_b
        W += delta_W
        b += delta_b

        self.previous_W = delta_W
        self.previous_b = delta_b
