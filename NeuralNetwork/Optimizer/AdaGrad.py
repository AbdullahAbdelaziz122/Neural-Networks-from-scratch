import numpy as np


class AdaGrad:
    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 / (1 + self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.biases_cache += layer.dbiases ** 2

        layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)


    def post_update(self):
        self.iterations +=1