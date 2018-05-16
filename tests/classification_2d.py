
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

from tests.tools.classification import *

from feedforward.regularizer import *

np.random.seed(1)

N = 500


# Ex 1. Two uniform segments
X = np.concatenate((np.random.rand(N, 1) * 2, np.random.rand(N, 1) * 2), axis=1)
D = np.zeros((N, 1))

for i in range(0, N):
    if X[i, 0] < 1:
        D[i] = 0
    else:
        D[i] = 1
X = rotate(X, 60)

network = Network(CrossEntropyLoss(), MomentumUpdater(learning_rate=0.5), L1(0.1))
network.add_layer(FullSigmoidLayer(1))

train_and_plot(network, epochs=500, batch_size=50, X=X, D=D, train_fraction=0.5)


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

train_and_plot(network, epochs=1000, batch_size=50, X=X, D=D, train_fraction=0.75)


# Ex 3. Spiral
spd = 5
scl = 6
t = np.arange(N//2).reshape((N//2,1)) / (N//2) + 0.1
x = np.concatenate((np.cos(t*np.pi*spd)*t*scl, np.cos(t*np.pi*spd + np.pi)*t*scl), axis=0)
y = np.concatenate((np.sin(t*np.pi*spd)*t*scl, np.sin(t*np.pi*spd + np.pi)*t*scl), axis=0)

X = np.concatenate((x, y), axis=1)
D = np.concatenate((np.zeros((N//2, 1)), np.ones((N//2, 1))))

network = Network(CrossEntropyLoss(), MomentumUpdater(learning_rate=0.03))
network.add_layer(FullSigmoidLayer(100))
network.add_layer(FullSigmoidLayer(1))

train_and_plot(network, epochs=2000, batch_size=50, X=X, D=D, train_fraction=0.75)


plt.show()


