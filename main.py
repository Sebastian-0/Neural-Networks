import numpy as np

from layer import *
from loss import *
from network import *
from weight_updater import *

# X = np.array([1, 2, 3, 4]).reshape((1, 4))
# D = np.array([1, 2]).reshape(1, 2)

X = np.array([1, 2, 3, 4,
              4, 3, 2, 1]).reshape((2, 4))
D = np.array([1, 2,
              1, 2]).reshape(2, 2)
# X = np.array([0, 0,
#               1, 1,
#               2, 2,
#               3, 3]).reshape((4, 2))
# D = np.array([0, 1, 2, 3]).reshape(4, 1)
print(X.shape)

# l = FullSigmoidLayer(3)
# l.init_weights(4)
# print(l.forward(X))
# loss = Loss(1)
# loss.init_weights(4)
# print(loss.loss(X, D))

network = Network()
network.add_layer(FullSigmoidLayer(3))
network.add_layer(FullLinearLayer(2))
network.set_loss(MeanSquareLoss())
network.set_weight_updater(SGDUpdater())

network.train(X, D, 100)

exit(0)


# Verify loss
print("Loss verification")
eps = 1e-4
Y = np.array([4.0, 5.0, 6.0]).reshape((1, 3))
# D = np.array([3.0, 2.0, 1.0]).reshape((1, 3))
D = np.array([3.0]).reshape((1, 1))
lin = FullLinearLayer(D.shape[1])
loss = MeanSquareLoss()

lin.init_weights(Y.shape[1], SGDUpdater())

print("Loss is: ")
print(loss.loss(lin.forward(Y), D))

# Derivative for weights
print("dLdW (this only works if you change the layer output to dLdW):")
for i in range(0, Y.shape[1]):
    lin.W[i, 0] -= eps
    Lm = loss.loss(lin.forward(Y), D)
    lin.W[i, 0] += 2*eps
    Lp = loss.loss(lin.forward(Y), D)
    lin.W[i, 0] -= eps
    num_der = (Lp - Lm) / (2*eps)
    print(num_der)

dLdy = loss.backward(lin.forward(Y), D)
print(lin.backward(dLdy))


# Derivative for inputs
print("dLdx")
for i in range(0, Y.shape[1]):
    Ym = np.copy(Y)
    Yp = np.copy(Y)
    Ym[0, i] -= eps
    Yp[0, i] += eps

    num_der = (loss.loss(lin.forward(Yp), D) - loss.loss(lin.forward(Ym), D)) / (2*eps)
    print(num_der)

dLdy = loss.backward(lin.forward(Y), D)
print(lin.backward(dLdy))