
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

x_max = 10
N = 1000

def train_and_plot(network, epochs, batch_size, function, N_train, noise_dev = 1):
    step_size = N // N_train
    if N % N_train != 0:
        step_size = N // (N_train - 1) - 1

    x_test = np.arange(0, x_max, x_max / N).reshape((N, 1))
    y_test = function(x_test)
    x_train = x_test[0::step_size]
    y_train = y_test[0::step_size] + np.random.normal(0, noise_dev, (N_train, 1))

    # Normalize output
    y_mean = y_test.mean(axis=0)
    y_std = y_test.std(axis=0)
    y_test = (y_test - y_mean) / y_std
    y_train = (y_train - y_mean) / y_std

    print(x_train.shape)
    print(y_train.shape)

    losses = network.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    plt.figure(figsize=(5.5, 7))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    # plt.axis([0, epochs, 0, max(losses)])
    ax1.plot(np.arange(epochs), losses, label="Loss (LSE)")
    # plt.legend()
    # plt.show()

    l, = ax2.plot(x_train, y_train, 'o', label="Training data")
    l.set_color(l.get_color() + "a0")

    ax2.plot(x_test, y_test, '--', label="True function")
    ax2.plot(x_test, network.predict(x_test), '-', label="Network prediction")

    # plt.tight_layout()
    plt.legend()
    plt.show(block=False)


# Ex 1 - Linear
network = Network(MeanSquareLoss(), MomentumUpdater(0.01))
# network.add_layer(FullSigmoidLayer(10))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=500, batch_size=10, function=(lambda x: x*2 + 10), N_train=10, noise_dev=1)


# Ex 2 - Quadratic
network = Network(MeanSquareLoss(), MomentumUpdater(0.03))
network.add_layer(FullSigmoidLayer(2))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=2000, batch_size=50, function=(lambda x: x**2), N_train=100, noise_dev=5)


# Ex 3 - Sine
network = Network(MeanSquareLoss(), MomentumUpdater(0.06))
network.add_layer(FullSigmoidLayer(15))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=3000, batch_size=100, function=(lambda x: np.sin(x*np.pi/3)), N_train=200, noise_dev=0.1)

plt.show()

