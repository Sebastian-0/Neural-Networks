
import numpy as np
from layer import Layer
from loss import Loss

class Network:
    def __init__(self):
        self.loss = loss
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

    def train(self, X, D):
        pass
        # TODO here!!!

