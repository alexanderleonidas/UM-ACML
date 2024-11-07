import numpy as np
from const import *
from neuralnet import NerualNet
from loss import Loss, MSE
import matplotlib.pyplot as plt

def plot_loss(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Mean Squared Error)")
    plt.title("Loss Over Time")
    plt.legend()
    plt.show()

def plot_weight_heatmaps(nn: NerualNet):
    # Visualize the learned weights as heatmaps.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the heatmap for weights_input_hidden
    im1 = axs[0].imshow(nn.network[0].weights, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Weights: Input to Hidden Layer")
    axs[0].set_xlabel("Hidden Neurons")
    axs[0].set_ylabel("Input Nodes")

    # Plot the heatmap for weights_hidden_output
    im2 = axs[1].imshow(nn.network[2].weights, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Weights: Hidden to Output Layer")
    axs[1].set_xlabel("Output Nodes")
    axs[1].set_ylabel("Hidden Neurons")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n=== Training Phase ===")
    # Now proceed with actual training
    x = np.eye(8)  # Real training data
    y = x.copy()
    
    nn = NerualNet()
    loss_func = MSE()
    
    plot_weight_heatmaps(nn)

    final_loss = nn.train(x, y, loss_func)

    plot_loss(nn.loss_history)
    plot_weight_heatmaps(nn)
    
    print(f"\nTraining completed with final loss: {final_loss}")

    # Test the trained network
    output = x
    for layer in nn.network:
        layer.forward(output)
        output = layer.output

    print("\nFinal loss:", loss_func.calculate(output, y))
    print("\nExample predictions:")
    for i in range(len(x)):
        print(f"Input {i}:")
        print("Target:", y[i])
        print("Prediction:", np.round(output[i]))
        print()

