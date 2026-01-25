import numpy as np
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, d_in):
        d_result = np.array(d_in, copy=True)
        d_result[self.inputs <= 0] = 0
        return d_result