import numpy as np


def weight(shape):
    return np.random.normal(loc=0, scale=3, size=shape)


def bias(shape):
    return np.random.normal(loc=0, scale=2, size=shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neural_network(x, weights, biases):
    h = x
    for i, w in enumerate(weights[:-1]):
        h = sigmoid(np.matmul(h, w) + biases[i])

    out = np.matmul(h, weights[-1]) + biases[-1]
    return np.reshape(out, [-1, 1])
