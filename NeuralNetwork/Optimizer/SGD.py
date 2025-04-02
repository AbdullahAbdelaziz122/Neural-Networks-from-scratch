import numpy as np
from NeuralNetwork.Layer import DenseLayer


class SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0):
        self.learning_rate = learning_rate
        self.current_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update_params(self, layer: DenseLayer):

        if self.momentum:

            if not hasattr(layer, "weight_momentums"):

                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weights_momentums - self.current_rate * layer.dweights
            layer.weights_momentums = weight_updates

            biases_updates = self.momentum * layer.biases_momentums - self.current_rate * layer.dbiases
            layer.biases_momentums = biases_updates

        else:
            weight_updates = -self.current_rate * layer.dweights
            biases_updates = -self.current_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += biases_updates

    def pre_update(self):
        if self.decay:
            self.current_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def post_update(self):
        self.iterations += 1
