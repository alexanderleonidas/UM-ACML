import numpy as np
from const import *
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * rng.random((n_inputs, n_neurons))
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
    
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)