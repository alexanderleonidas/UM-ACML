from neuralnetwork import NeuralNetwork
import numpy as np

# Initialize network
input_size = 8
hidden_size = 3
output_size = 8
learning_rate = 0.1

# Define the training data
training_inputs = np.eye(input_size)  # Identity matrix for 8 input patterns
training_outputs = np.eye(output_size)  # Target output matches input

# Create and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
nn.train(training_inputs, training_outputs, epochs=10000)

# Test the trained network
print("\nTesting the trained network:")
for i, test_input in enumerate(training_inputs):
    predicted_output = nn.predict(test_input.reshape(1, -1))
    print(f"Input: {test_input} Predicted Output: {np.round(predicted_output.flatten())}")

# Gradient check
print("\nPerforming gradient check:")
nn.gradient_check(training_inputs, training_outputs)

# Plot loss
print("\nVisualising loss:")
nn.plot_loss()

# Plot weight heatmaps
print("\nVisualizing weight heatmaps:")
nn.plot_weight_heatmaps()