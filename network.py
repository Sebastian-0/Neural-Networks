
import numpy as np
from layer import Layer
from loss import Loss

import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.loss = None
        self.weight_updater = None
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def set_weight_updater(self, updater):
        self.weight_updater = updater

    def init(self, input_dimension):
        size = self.layers[0].n_nodes
        self.layers[0].init_weights(input_dimension, self.weight_updater)
        for l in self.layers[1:]:
            l.init_weights(size, self.weight_updater)
            size = l.n_nodes

    def train(self, X, D, epochs):
        N = X.shape[1]
        self.init(N)

        losses = []
        for iter in range(0, epochs):
            # Forward propagation
            out = X
            for l in self.layers:
                out = l.forward(out)

            L = self.loss.loss(out, D)
            losses.append(L)

            dLdy = self.loss.backward(out, D)
            for l in reversed(self.layers):
                dLdy = l.backward(dLdy)

        plt.plot(np.arange(epochs), losses)
        plt.show()
        # TODO Display loss as graph?

