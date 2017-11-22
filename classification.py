
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

N = 500
X = np.concatenate((np.random.rand(N, 1) * 2, np.random.rand(N, 1) * 2), axis=1)
D = np.zeros((N, 1))

for i in range(0, N):
    if X[i, 0] < 1:
        if X[i, 1] < 1:
            D[i] = 0
        else:
            D[i] = 1
    else:
        if X[i, 1] < 1:
            D[i] = 1
        else:
            D[i] = 0

X -= 1
crosses = np.where(D == 0)[0]
dots = np.where(D != 0)[0]

# plt.axis([0, epochs, 0, max(losses)])
plt.plot(X[crosses, 0], X[crosses, 1], 'ro')
plt.plot(X[dots, 0], X[dots, 1], 'bo')
plt.show()


network = Network()
network.add_layer(FullSigmoidLayer(8))
network.add_layer(FullSigmoidLayer(1))
network.set_loss(MeanSquareLoss())
network.set_weight_updater(SGDUpdater())

network.train(X, D, 1000)


