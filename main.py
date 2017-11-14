import numpy as np

from layer import *

l = FullSigmoidLayer(3)
l.init_weights(4)

X = np.array([1, 2, 3, 4])

print(l.forward(X))
