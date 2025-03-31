import numpy as np
from NeuralNetwork.Layer import DenseLayer


class SGD:
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_params(self, layer: DenseLayer):
        layer.weights += - self.current_rate * layer.dweights
        layer.biases += - self.current_rate * layer.dbiases

    def pre_update(self):
        if self.decay:
            self.current_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def post_update(self):
        self.iterations += 1
