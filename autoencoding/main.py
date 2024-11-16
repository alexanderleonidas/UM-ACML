import torch.nn as nn
import torch.optim as optim
from simple_autoencoder import SimpleAutoencoder
from fewer_channels_autoencoder import FewerChannelsAutoencoder
from reduced_autoencoder import ReducedAutoencoder
from lrare_latent_autoencoder import LargerLatentSpaceAutoencoder
from utils import *
from const import *

# ---------------------------------- #
# ----------- EXERCISE 1 ----------- #
# ---------------------------------- #

# Load a single batch for visualization
batch_file = 'data_batch_1' # Change the batch file as needed
train_loader, val_loader, test_loader = load_cifar_batch(batch_file)

# Instantiate and set up the model
autoencoder = SimpleAutoencoder().to(device)
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
train_loss_history, val_loss_history = train_autoencoder(autoencoder, optimizer, criterion, train_loader, val_loader)

# Plot the loss and validation loss
plot_loss(train_loss_history, 'Simple Autoencoder Training Loss')
plot_loss(val_loss_history, 'Simple Autoencoder Validation Loss')

# Evaluate on test set
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = autoencoder(inputs)
        test_loss += criterion(outputs, inputs).item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# Visualize the reconstruction of input images (optional)
visualize_reconstruction(autoencoder, test_loader)

# ---------------------------------- #
# ----------- EXERCISE 2 ----------- #
# ---------------------------------- #

# Load a single batch for visualization
batch_file = 'data_batch_2' # Change the batch file as needed
train_loader, val_loader, test_loader = load_cifar_batch(batch_file)

models = [
    FewerChannelsAutoencoder().to(device),
    ReducedAutoencoder().to(device),
    LargerLatentSpaceAutoencoder().to(device)
]
criterion = nn.MSELoss()  # Mean squared error loss
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

train_loss_history = []
val_loss_history = []
test_loss_history = []

for model, optimizer in zip(models, optimizers):
    loss, val_loss = train_autoencoder(model, optimizer, criterion, train_loader, val_loader)
    train_loss_history.append(loss)
    val_loss_history.append(val_loss)

    # Evaluate on test set
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, inputs).item()

    test_loss_history.append(test_loss / len(test_loader))

# Plot loss
model_names = ['FewerChannelsAutoencoder', 'ReducedAutoencoder', 'LargerLatentSpaceAutoencoder']
for i in range(3):
    plot_loss(train_loss_history[i], model_names[i] + 'Training Loss')
    plot_loss(val_loss_history[i], model_names[i] + 'Validation Loss')