import matplotlib.pyplot as plt
import numpy as np
import time

class Network:
    def __init__(self, loss, weight_updater):
        self.loss = loss
        self.weight_updater = weight_updater
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def init(self, input_dimension):
        size = self.layers[0].n_nodes
        self.layers[0].init_weights(input_dimension, self.weight_updater)
        for l in self.layers[1:]:
            l.init_weights(size, self.weight_updater)
            size = l.n_nodes

    def train(self, X, D, epochs, batch_size=50, validation_data=None):
        assert self.layers, "The network contains no layers!"
        assert self.weight_updater, "The network has no weight updater!"
        assert self.loss, "The network has no loss function!"
        assert (X.shape[0] == D.shape[0]), "#input != #targets: {0} != {1}".format(X.shape[0], D.shape[0])
        assert (D.shape[1] == self.layers[-1].n_nodes), "Dimension of network output and target data does not agree: {0} != {1}".format(self.layers[-1].n_nodes, D.shape[1])

        if validation_data:
            V_X, V_D = validation_data

        start = time.time()

        P = X.shape[0]
        N = X.shape[1]
        self.init(N)

        I = np.arange(P)
        batch_size = min(batch_size, P)

        training_losses = []
        validation_losses = []
        for iter in range(epochs):
            np.random.shuffle(I)
            for i in range(0, P, batch_size):
                if i + batch_size > P:  # Don't run on a small number of samples
                    break

                if i + batch_size*2 > P:  # If there are too few points remaining include them as well
                    Ib = I[i:]
                else:
                    Ib = I[i:i+batch_size]

                # Select samples
                Xb = X[Ib, :]
                Db = D[Ib, :]

                # Forward propagation
                out = Xb
                for l in self.layers:
                    out = l.forward(out)

                # Backwards propagation
                dLdy = self.loss.backward(out, Db)
                for l in reversed(self.layers):
                    dLdy = l.backward(dLdy)

            training_losses.append(self.loss_for(X, D))
            if validation_data:
                validation_losses.append(self.loss_for(V_X, V_D))

        end = time.time()
        print("Time: %f" % (end - start))

        return training_losses, validation_losses
        # TODO Display loss as graph in realtime?

    def predict(self, X):
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out

    def loss_for(self, X, D):
        # Forward propagation
        out = X
        for l in self.layers:
            out = l.forward(out)

        return self.loss.loss(out, D)
