
from feedforward.loss import *
from feedforward.network import *
from feedforward.weight_updater import *
from feedforward.regularizer import *

from feedforward.layer import *

import numpy as np
import matplotlib.pyplot as plt

from tests.tools.regression import *

np.random.seed(1)


# Ex 1 - Linear
network = Network(MeanSquareLoss(), MomentumUpdater(0.05), L1(0.1))
network.add_layer(FullSigmoidLayer(10))
network.add_layer(FullLinearLayer(1))

train_and_plot(network, epochs=500, batch_size=10, function=(lambda x: x*2 + 10), train_fraction=10/100, noise_dev=0.5, N=100)

#
# # Ex 2 - Quadratic
# network = Network(MeanSquareLoss(), MomentumUpdater(0.03))
# network.add_layer(FullSigmoidLayer(2))
# network.add_layer(FullLinearLayer(1))
#
# train_and_plot(network, epochs=2000, batch_size=50, function=(lambda x: x**2), train_fraction=1/10)
#
#
# # Ex 3 - Sine
# network = Network(MeanSquareLoss(), MomentumUpdater(0.06))
# network.add_layer(FullSigmoidLayer(15))
# network.add_layer(FullLinearLayer(1))
#
# train_and_plot(network, epochs=3000, batch_size=100, function=(lambda x: np.sin(x*np.pi/3)), train_fraction=2/10)

plt.show()

