
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

x_max = 10
N = 1000
x = np.arange(0, x_max, x_max/N).reshape((N, 1))
y_pure = x * 2 + 10
y = y_pure + np.random.normal(0, 1, (N, 1))

print(x.shape)
print(y.shape)

network = Network()
# network.add_layer(FullLinearLayer(3))
network.add_layer(FullLinearLayer(1))
network.set_loss(MeanSquareLoss())
network.set_weight_updater(SGDUpdater())

network.train(x, y, 400)

print("Network props")
print(network.layers[0].W)
print(network.layers[0].b)

plt.plot(x, y, label="Training data")
plt.plot(x, y_pure, label="True function")
plt.plot(x, network.predict(x), label="Network prediction")
plt.legend()
plt.show()
