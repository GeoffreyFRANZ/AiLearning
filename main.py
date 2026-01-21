import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
# X = [[1, 2 , 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]
X, y =  spiral_data(100, 3)

class Layer_Dense:
    def __init__(self,  n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,  self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, d_in):
        d_result = np.array(d_in, copy=True)
        d_result[self.inputs <= 0] = 0
        return d_result

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def backward(self, dvalues):
        self.dinputs =  np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss =  np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 1:
            dvalues = dvalues.copy()
            dvalues[range(samples), y_true] -= 1
        elif len(y_true.shape) == 2:
            dvalues = dvalues.copy()
            dvalues -= y_true
        dvalues /= samples
        return dvalues

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return np.mean(self.loss.forward(self.output, y_true))
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.d_inputs = dvalues.copy()
        self.d_inputs[range(samples), y_true] -= 1
        self.d_inputs = self.d_inputs / samples
        return self.d_inputs

class Optimizer_SGD:
    def __init__(self, learning_rate= 1., decay= 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epoch = 0
    def pre_update_params(self):
        self.current_learning_rate = (self.learning_rate *
                                      (1. / (1. + self.decay * self.epoch)))
    def update_params(self, layer):
        layer.weights -= self.current_learning_rate * layer.dweights
        layer.biases  -= self.current_learning_rate * layer.dbiases
    def post_update_params(self):
        self.epoch += 1


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