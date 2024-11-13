import torch.nn as nn
import torch.optim as optim
from autoencoder import Autoencoder
from utils import *
from const import *


# Load a single batch for visualization
batch_file = 'data_batch_1' # Change the batch file as needed
train_loader, val_loader, test_loader = load_cifar_batch(batch_file)

# Instantiate and set up the model
autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
train_loss_history = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)  # Compare reconstructed output to the input
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Plot the Loss
plot_loss(train_loss_history)

# Evaluate on test set (optional)
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = autoencoder(inputs)
        test_loss += criterion(outputs, inputs).item()

print(f"Test Loss: {test_loss / len(test_loader):.4f}")
