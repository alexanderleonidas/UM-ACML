import numpy as np

class Activation:
    def forward(self, x):
        self.output =  1 / (1 + np.exp(-x))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)