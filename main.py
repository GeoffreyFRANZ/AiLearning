import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from classes import (Layer_Dense, Activation_Softmax,
                     Activation_ReLU, Optimizer_SGD,
                     Activation_Softmax_Loss_CategoricalCrossentropy,
                     Optimizer_Adagrad, Optimizer_RMSProp
                     )

nnfs.init()
X, y =  spiral_data(100, 3)
layer1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(64, 3)
loss_activation_function = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_RMSProp(decay = 1e-4)
for epoch in range(10001):
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation_function.forward(layer2.output, y)
    predictions = np.argmax(loss_activation_function.output, axis=1)
    if len (y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}'
              )
    loss_activation_function.backward(loss_activation_function.output, y)
    layer2.backward(loss_activation_function.d_inputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()