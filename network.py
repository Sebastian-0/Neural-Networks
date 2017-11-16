
import numpy as np
from layer import Layer
from loss import Loss

class Network:
    def __init__(self):
        self.loss = None
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def init(self, input_dimension):
        size = self.layers[0].n_nodes
        self.layers[0].init_weights(input_dimension)
        for l in self.layers[1:]:
            l.init_weights(size)
            size = l.n_nodes

    def train(self, X, D, epochs):
        N = X.shape[1]
        self.init(N)

        for _ in range(0, epochs):
            # Forward propagation
            out = X
            for l in self.layers:
                out = l.forward(out)

            L = self.loss.loss(out, D)
            print(L)

            dLdy = self.loss.backward(out, D)
            print(dLdy)

            dLdh = self.layers[-1].backward(dLdy)
            print(dLdh)

        # TODO here!!!

