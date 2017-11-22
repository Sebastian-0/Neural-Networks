
import numpy as np

class WeightUpdater(object):
    def update(self, W, dLdW):
        raise NotImplementedError


class SGDUpdater(WeightUpdater):
    def __init__(self):
        self.learning_rate = 0.01
        self.batch_size = 50

    def update(self, W, dLdW):
        W -= self.learning_rate * dLdW

    def select_samples(self, X, D):
        N = X.shape[0]
        I = np.random.randint(0, N, self.batch_size)

        Xb = X[I, :]
        Db = D[I, :]
        return Xb, Db
