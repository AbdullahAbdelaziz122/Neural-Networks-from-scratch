import numpy as np
from Activation import ReLu, Softmax
from Datasets import DataGeneration
from Loss import CategoricalCrossEntropy
from Optimizer import SGD
from Layer import DenseLayer

# Data
np.random.seed(42)
X, y_true = DataGeneration.spiral_data(100, 3)


# Neural Network initializing
dense1 = DenseLayer.LayerDense(2, 64)
relu = ReLu.ReLu()

dense2 = DenseLayer.LayerDense(64, 3)
softmax = Softmax.Softmax()
loss = CategoricalCrossEntropy.CategoricalCrossEntropy()

optimizer = SGD.SGD(learning_rate=1, decay=1e-3 )

# Training
epochs = 10001
if len(y_true) == 2:
    y_true = np.argmax(y_true, axis=1)


for epoch in range(epochs):

    # Forward Pass
    dense1.forward(X)
    relu.forward(dense1.output)

    dense2.forward(relu.output)
    softmax.forward(dense2.output)

    loss_value = loss.calculate(softmax.output, y_true)

    # Calculate accuracy
    predictions = np.argmax(softmax.output, axis=1)

    accuracy = np.mean(predictions == y_true)

    # Print progress every 100 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Acc: {accuracy:.3f}, Loss: {loss_value:.3f}")

    # Backward pass
    loss.backward(softmax.output, y_true)
    softmax.backward(loss.dinputs)
    dense2.backward(softmax.dinputs)
    relu.backward(dense2.dinputs)
    dense1.backward(relu.dinputs)


    # Update params
    optimizer.pre_update()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update()






