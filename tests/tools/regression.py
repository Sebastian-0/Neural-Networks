import numpy as np
import matplotlib.pyplot as plt

x_max = 10

def train_and_plot(network, epochs, batch_size, function, train_fraction, N=1000, noise_dev = 0.1):
    N_train = int(N * train_fraction)
    I = np.arange(N)
    np.random.shuffle(I)

    X = np.arange(0, x_max, x_max / N).reshape((N, 1))
    D = function(X)

    # Normalize output
    y_mean = D.mean(axis=0)
    y_std = D.std(axis=0)
    D = (D - y_mean) / y_std

    y_true = np.copy(D)
    D += np.random.normal(0, noise_dev, (N, 1))

    x_train = X[I[0:N_train], ]
    y_train = D[I[0:N_train]]
    x_test = X[I[N_train:], ]
    y_test = D[I[N_train:]]

    print(x_train.shape)
    print(y_train.shape)

    training_losses, validation_losses = network.train(x_train, y_train,
                                                       epochs=epochs,
                                                       batch_size=batch_size,
                                                       validation_data=(x_test, y_test))

    plt.figure(figsize=(5.5, 7))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    # plt.axis([0, epochs, 0, max(losses)])
    ax1.plot(np.arange(epochs), training_losses, label="Training Loss")
    ax1.plot(np.arange(epochs), validation_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error")
    ax1.legend()

    l1, = ax2.plot(x_train, y_train, 'o', label="Training data")
    l1.set_color(l1.get_color() + "a0")
    # l2, = ax2.plot(x_test, y_test, 'o', label="Test data")
    # l2.set_color(l2.get_color() + "a0")

    ax2.plot(X, y_true, '--', label="True function")
    ax2.plot(X, network.predict(X), '-', label="Network prediction")

    plt.legend()
    plt.show(block=False)