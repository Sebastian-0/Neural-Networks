import numpy as np
import matplotlib.pyplot as plt

N_boundary_points = 300

def rotate(X, angle_in_degrees):
    theta = np.radians(angle_in_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])  # rotation matrix
    return np.dot(X, R)


def train_and_plot(network, epochs, batch_size, X, D, train_fraction):
    N_train = int(X.shape[0] * train_fraction)
    I = np.arange(X.shape[0])
    np.random.shuffle(I)

    # y_mean = X.mean(axis=0)
    # y_std = X.std(axis=0)
    # X = (X - y_mean) / y_std

    x_train = X[I[0:N_train], ]
    d_train = D[I[0:N_train]]
    x_test = X[I[N_train:], ]
    d_test = D[I[N_train:]]

    training_losses, validation_losses = network.train(x_train, d_train,
                                                       epochs=epochs,
                                                       batch_size=batch_size,
                                                       validation_data=(x_test, d_test))

    plt.figure(figsize=(5.3, 7.5))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(np.arange(epochs), training_losses, label="Training Loss")
    ax1.plot(np.arange(epochs), validation_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error")
    ax1.legend()

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