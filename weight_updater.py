
class WeightUpdater(object):
    def update(self, W, dLdW):
        raise NotImplementedError


class SGDUpdater(WeightUpdater):
    def __init__(self):
        self.learning_rate = 0.01

    def update(self, W, dLdW):
        W -= self.learning_rate * dLdW