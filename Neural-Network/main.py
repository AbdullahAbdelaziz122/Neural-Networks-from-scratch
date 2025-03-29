from Activations import *
from DataGeneration import *
from DenseLayer import LayerDense
from Loss import CategoricalCrossEntropy

X, y = spiral_data(points=100, classes=3)
print(X.shape, y.shape)

dense1 = LayerDense(2, 10)
relu1 = ReLu()

dense2 = LayerDense(10, 10)
relu2 = ReLu()

dense3 = LayerDense(10, 3)
softmax = Softmax()

# Training feedforward
dense1.forward(X)
relu1.forward(dense1.output)

dense2.forward(relu1.output)
relu2.forward(dense2.output)

dense3.forward(relu2.output)
softmax.forward(dense3.output)

output = softmax.output

loss = CategoricalCrossEntropy()
loss_value = loss.calculate(output, y)


# Accuracy
def accuracy(softmax_output, y_true):
    predictions = np.argmax(softmax_output, axis=1)
    return np.mean(predictions == y_true)


acc = accuracy(output, y)

print("Categorical Cross-Entropy Loss:", loss_value)
print(f"acc: {acc.round(2) * 100}%")
