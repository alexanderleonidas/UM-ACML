import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initialize()
    
    def initialize(self):
        rng = np.random.default_rng(42)
        self.weights_input_hidden = rng.random((self.input_size, self.hidden_size)) - 0.5
        self.bias_hidden = rng.random((1, self.hidden_size)) - 0.5
        self.weights_hidden_output = rng.random((self.hidden_size, self.output_size)) - 0.5
        self.bias_output = rng.random((1, self.output_size)) - 0.5
    
    def forward_propagate(self, inputs):
        # Calculate activations for the hidden layer
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._sigmoid(self.hidden_input)

        # Calculate activations for the output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._sigmoid(self.final_input)

        return self.final_output
    
    def backpropagate(self, inputs, target_outputs):
        # Calculate the output error
        output_error = target_outputs - self.final_output
        d_output = output_error * self._sigmoid_derivative(self.final_output)

        # Calculate error for the hidden layer
        hidden_error = np.dot(d_output, self.weights_hidden_output.T)
        d_hidden = hidden_error * self._sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, d_output)
        self.bias_output += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights_input_hidden += self.learning_rate * np.dot(inputs.T, d_hidden)
        self.bias_hidden += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

        return np.mean(np.square(output_error))  # Mean Squared Error for loss
    
    def train(self, inputs, target_outputs, epochs=10000):
        for epoch in range(epochs):
            # Forward and backpropagation in a single batch
            self.forward_propagate(inputs)
            loss = self.backpropagate(inputs, target_outputs)

            # Print loss every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} Loss: {loss}")

    def predict(self, inputs):
        # Perform forward propagation to predict outputs
        predictions = self.forward_propagate(inputs)
        return predictions

    # Sigmoid activation function and its derivative
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)