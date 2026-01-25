import numpy as np
class Loss:
    def __init__(self):
        pass
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss =  np.mean(sample_losses)
        return data_loss