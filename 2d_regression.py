
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

x_max = 10
N = 1000

def train_and_plot(network, epochs, batch_size, function, train_fraction, noise_dev = 0.1):
    N_train = int(N * train_fraction)
    I = np.arange(N)
    np.random.shuffle(I)

    X = np.arange(0, x_max, x_max / N).reshape((N, 1))
    D = function(X)

    # Normalize output
    y_mean = D.mean(axis=0)
    y_std = D.std(axis=0)
    D = (D - y_mean) / y_std

    y_true = np.copy(D)
    D += np.random.normal(0, noise_dev, (N, 1))

    x_train = X[I[0:N_train], ]
    y_train = D[I[0:N_train]]
    x_test = X[I[N_train:], ]
    y_test = D[I[N_train:]]

    print(x_train.shape)
    print(y_train.shape)

    training_losses, validation_losses = network.train(x_train, y_train,
                                                       epochs=epochs,
                                                       batch_size=batch_size,
                                                       validation_data=(x_test, y_test))

    plt.figure(figsize=(5.5, 7))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    # plt.axis([0, epochs, 0, max(losses)])
    ax1.plot(np.arange(epochs), training_losses, label="Training Loss")
    ax1.plot(np.arange(epochs), validation_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error")
    ax1.legend()

    l1, = ax2.plot(x_train, y_train, 'o', label="Training data")
    l1.set_color(l1.get_color() + "a0")
    # l2, = ax2.plot(x_test, y_test, 'o', label="Test data")
    # l2.set_color(l2.get_color() + "a0")

    ax2.plot(X, y_true, '--', label="True function")
    ax2.plot(X, network.predict(X), '-', label="Network prediction")

    plt.legend()
    plt.show(block=False)


# Ex 1 - Linear
network = Network(MeanSquareLoss(), MomentumUpdater(0.01))
# network.add_layer(FullSigmoidLayer(10))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=500, batch_size=10, function=(lambda x: x*2 + 10), train_fraction=1/100)


# Ex 2 - Quadratic
network = Network(MeanSquareLoss(), MomentumUpdater(0.03))
network.add_layer(FullSigmoidLayer(2))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=2000, batch_size=50, function=(lambda x: x**2), train_fraction=1/10)


# Ex 3 - Sine
network = Network(MeanSquareLoss(), MomentumUpdater(0.06))
network.add_layer(FullSigmoidLayer(15))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=3000, batch_size=100, function=(lambda x: np.sin(x*np.pi/3)), train_fraction=2/10)

plt.show()

