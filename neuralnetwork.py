import numpy as np
import matplotlib.pyplot as plt

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

        # Gradients for weights and biases
        self.grad_weights_hidden_output = np.dot(self.hidden_output.T, d_output)
        self.grad_bias_output = np.sum(d_output, axis=0, keepdims=True)
        self.grad_weights_input_hidden = np.dot(inputs.T, d_hidden)
        self.grad_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * self.grad_weights_hidden_output
        self.bias_output += self.learning_rate * self.grad_bias_output
        self.weights_input_hidden += self.learning_rate * self.grad_weights_input_hidden
        self.bias_hidden += self.learning_rate * self.grad_bias_hidden

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

    def gradient_check(self, inputs, target_outputs, epsilon=1e-5):
        # Perform a numerical gradient check for the weights of the network.

        # Forward propagate and backpropagate to get gradients from the network
        self.forward_propagate(inputs)
        self.backpropagate(inputs, target_outputs)
        
        # Check gradients for weights_input_hidden
        print("Checking gradients for weights_input_hidden...")
        for i in range(self.weights_input_hidden.shape[0]):
            for j in range(self.weights_input_hidden.shape[1]):
                # Perturb weight by a small amount epsilon
                original_value = self.weights_input_hidden[i, j]
                self.weights_input_hidden[i, j] = original_value + epsilon
                loss_plus = np.mean((self.forward_propagate(inputs) - target_outputs) ** 2)
                
                self.weights_input_hidden[i, j] = original_value - epsilon
                loss_minus = np.mean((self.forward_propagate(inputs) - target_outputs) ** 2)
                
                # Restore original weight value
                self.weights_input_hidden[i, j] = original_value
                
                # Calculate numerical gradient
                numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Compare with backprop gradient
                backprop_gradient = self.grad_weights_input_hidden[i, j]
                difference = np.abs(numerical_gradient - backprop_gradient)
                
                if difference > 1e-4:
                    print(f"Gradient check failed at ({i}, {j}):")
                    print(f"Numerical: {numerical_gradient}, Backprop: {backprop_gradient}, Difference: {difference}")
                else:
                    print(f"Gradient check passed at ({i}, {j}).")

    def plot_weight_heatmaps(self):
        # Visualize the learned weights as heatmaps.
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the heatmap for weights_input_hidden
        axs[0].imshow(self.weights_input_hidden, cmap='viridis', aspect='auto')
        axs[0].set_title("Weights: Input to Hidden Layer")
        axs[0].set_xlabel("Hidden Neurons")
        axs[0].set_ylabel("Input Nodes")

        # Plot the heatmap for weights_hidden_output
        axs[1].imshow(self.weights_hidden_output, cmap='viridis', aspect='auto')
        axs[1].set_title("Weights: Hidden to Output Layer")
        axs[1].set_xlabel("Output Nodes")
        axs[1].set_ylabel("Hidden Neurons")

        plt.tight_layout()
        plt.show()

    # Sigmoid activation function and its derivative
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)