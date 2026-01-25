import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from classes import ( Layer_Dense, Activation_Softmax,
                     Activation_ReLU,Optimizer_SGD,
                     Activation_Softmax_Loss_CategoricalCrossentropy,
                     )

nnfs.init()
X, y =  spiral_data(100, 3)
layer1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
layer2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()
loss_activation_function = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay = 1e-2)
for epoch in range(10001):
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation_function.forward(layer2.output, y)

    predictions = np.argmax(loss_activation_function.output, axis=1)
    accuracy = np.mean(predictions == y)

    dvalues = loss_activation_function.backward(loss_activation_function.output, y)
    dvalues = layer2.backward(dvalues)
    dvalues = activation1.backward(dvalues)
    layer1.backward(dvalues)


    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')