import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        rng = np.random.default_rng(42)
        self.weights = rng.random((n_inputs, n_neurons))
        self.bias = rng.random((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias