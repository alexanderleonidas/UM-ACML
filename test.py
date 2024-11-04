# JUST A TEMPLATE DO NOT USE AS FINAL COPY.

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define neural network parameters
input_size = 8
hidden_size = 3
output_size = 8
learning_rate = 0.1

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
bias_hidden = np.random.rand(hidden_size) - 0.5
weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
bias_output = np.random.rand(output_size) - 0.5

# Define training data (8 unique examples)
training_inputs = np.eye(8)  # Identity matrix for the 8 possible input patterns
training_outputs = np.eye(8)  # Target output is the same as input

# Training loop
epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for idx in range(8):  # Train on all examples in each epoch
        input_layer = training_inputs[idx]
        target_output = training_outputs[idx]

        # Forward propagation
        hidden_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        final_output = sigmoid(final_input)

        # Calculate the error
        output_error = target_output - final_output
        total_loss += np.sum(output_error ** 2)

        # Backward propagation
        d_output = output_error * sigmoid_derivative(final_output)
        error_hidden_layer = d_output.dot(weights_hidden_output.T)
        d_hidden = error_hidden_layer * sigmoid_derivative(hidden_output)

        # Update weights and biases
        weights_hidden_output += learning_rate * np.outer(hidden_output, d_output)
        bias_output += learning_rate * d_output

        weights_input_hidden += learning_rate * np.outer(input_layer, d_hidden)
        bias_hidden += learning_rate * d_hidden

    # Print progress
    if epoch % 1000 == 0:
        avg_loss = total_loss / 8  # Average loss across all examples
        print(f"Epoch {epoch} Loss: {avg_loss:.6f}")

# Test the trained network
print("\nTesting trained network:")
for i in range(8):
    input_layer = training_inputs[i]
    hidden_output = sigmoid(np.dot(input_layer, weights_input_hidden) + bias_hidden)
    final_output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)
    final_output = np.round(final_output)  # Round output for binary results
    print(f"Input: {input_layer} Predicted Output: {final_output} Target: {training_outputs[i]}")
