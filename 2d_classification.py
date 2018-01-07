
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N = 200
N_boundary_points = 300

def rotate(X, angle_in_degrees):
    theta = np.radians(angle_in_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])  # rotation matrix
    return np.dot(X, R)


def train_and_plot(network, epochs, batch_size, X, D, N_train):
    I = np.arange(N)
    np.random.shuffle(I)

    x_test = X[I,]
    d_test = D[I]
    x_train = x_test[0:N_train,]
    d_train = d_test[0:N_train]

    losses = network.train(x_train, d_train, epochs=epochs, batch_size=batch_size)

    plt.figure(figsize=(5.3, 7.5))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(np.arange(epochs), losses, label="Loss (LSE)")

    x_min, x_max = x_test[:, 0].min() - .2, x_test[:, 0].max() + .2
    y_min, y_max = x_test[:, 1].min() - .2, x_test[:, 1].max() + .2
    # grid stepsize
    h = max(x_max - x_min, y_max - y_min) / N_boundary_points

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = network.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z[Z > 0.5] = 1
    Z[Z <= 0.5] = 0

    x_plot = x_test
    d_plot = d_test

    crosses = np.where(d_plot == 0)[0]
    dots = np.where(d_plot != 0)[0]

    ax2.plot(x_plot[crosses, 0], x_plot[crosses, 1], 'ro')
    ax2.plot(x_plot[dots, 0], x_plot[dots, 1], 'bo')
    ax2.contour(xx, yy, Z, cmap=plt.cm.Paired)
    ax2.axis([x_min, x_max, y_min, y_max])
    plt.show(block=False)


# Ex 1. Two uniform segments
X = np.concatenate((np.random.rand(N, 1) * 2, np.random.rand(N, 1) * 2), axis=1)
D = np.zeros((N, 1))

for i in range(0, N):
    if X[i, 0] < 1:
        D[i] = 0
    else:
        D[i] = 1

X = rotate(X, 30)

network = Network(CrossEntropyLoss(), MomentumUpdater(learning_rate=0.5))
network.add_layer(FullSigmoidLayer(1))

train_and_plot(network, epochs=500, batch_size=50, X=X, D=D, N_train=100)


# Ex 2. Uniform Checkerboard
X = np.concatenate((np.random.rand(N, 1) * 2, np.random.rand(N, 1) * 2), axis=1)
D = np.zeros((N, 1))

for i in range(0, N):
    if X[i, 0] < 1:
        if X[i, 1] < 1:
            D[i] = 0
            X[i, 1] -= 0.1
        else:
            X[i, 1] += 0.1
            D[i] = 1
        X[i, 0] -= 0.1
    else:
        if X[i, 1] < 1:
            D[i] = 1
            X[i, 1] -= 0.1
        else:
            D[i] = 0
            X[i, 1] += 0.1
        X[i, 0] += 0.1

network = Network(CrossEntropyLoss(), MomentumUpdater(learning_rate=0.5))
network.add_layer(FullSigmoidLayer(5))
network.add_layer(FullSigmoidLayer(1))

train_and_plot(network, epochs=1000, batch_size=50, X=X, D=D, N_train=800)


# Ex 3. Spiral
spd = 5
scl = 6
t = np.arange(N//2).reshape((N//2,1)) / (N//2) + 0.1
x = np.concatenate((np.cos(t*np.pi*spd)*t*scl, np.cos(t*np.pi*spd + np.pi)*t*scl), axis=0)
y = np.concatenate((np.sin(t*np.pi*spd)*t*scl, np.sin(t*np.pi*spd + np.pi)*t*scl), axis=0)

print(x.shape)
print(y.shape)

X = np.concatenate((x, y), axis=1)
D = np.concatenate((np.zeros((N//2, 1)), np.ones((N//2, 1))))

network = Network(CrossEntropyLoss(), MomentumUpdater(learning_rate=0.08))
network.add_layer(FullSigmoidLayer(100))
network.add_layer(FullSigmoidLayer(1))

train_and_plot(network, epochs=2000, batch_size=50, X=X, D=D, N_train=200)


plt.show()


