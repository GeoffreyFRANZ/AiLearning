import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
# X = [[1, 2 , 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]
X, y =  spiral_data(100, 3)

class Layer_Dense:
    #initialize weights and biases
    def __init__(self,  n_inputs, n_neurons):
        # weights  equal 0.10 * between numbers of inputs and neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # add  zeros  at least one to   n_neurons
        self.biases = np.zeros((1, n_neurons))
    #calculate output answer
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,  self.weights) + self.biases

    def backward(self, d_out):
        self.dweights = np.dot(self.inputs.T, d_out)
        self.dbiases = np.sum(d_out, axis=0, keepdims=True)
        self.dinputs = np.dot(d_out, self.weights.T)

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

        # Si y_true contient des labels entiers
        if len(y_true.shape) == 1:
            # Copier les prédictions
            dvalues = dvalues.copy()
            # Soustraire 1 à la bonne classe
            dvalues[range(samples), y_true] -= 1

        # Si y_true est one-hot
        elif len(y_true.shape) == 2:
            dvalues = dvalues.copy()
            dvalues -= y_true

        # Normaliser
        dvalues /= samples

        return dvalues

class Optimizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases  -= self.learning_rate * layer.dbiases


# --- Création des couches ---
layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.1)

# --- Boucle d'entraînement ---
for epoch in range(1000001):

    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Calcul de la loss
    loss = loss_function.calculate(activation2.output, y)

    # Calcul de l'accuracy (juste pour voir si ça marche)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass
    dvalues = loss_function.backward(activation2.output, y)
    dvalues = layer2.backward(dvalues)
    dvalues = activation1.backward(dvalues)
    dvalues = layer1.backward(dvalues)

    # Update des poids
    optimizer.update_params(layer2)
    optimizer.update_params(layer1)

    # Affichage toutes les 1000 epochs
    if epoch % 1000 == 0:
        print(f"epoch {epoch}, loss {loss:.4f}, accuracy {accuracy:.4f}")