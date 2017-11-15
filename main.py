import numpy as np

from layer import *
from loss import *
from network import *


X = np.array([1, 2, 3, 4,
              4, 3, 2, 1]).reshape((2,4))
D = np.array([1, 2]).reshape(2, 1)
print(X.shape)

# l = FullSigmoidLayer(3)
# l.init_weights(4)
# print(l.forward(X))
# loss = Loss(1)
# loss.init_weights(4)
# print(loss.loss(X, D))

network = Network()
network.add_layer(FullSigmoidLayer(3))
network.set_loss(Loss(2))

network.train(X, D, 1)