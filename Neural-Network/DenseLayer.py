import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = 0

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
