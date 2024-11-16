import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,  random_split, TensorDataset
from const import *

# Helper function to load CIFAR-10 batch files, normalise and split into training and testing
def load_cifar_batch(batch):
    file_path = os.path.join(DATA_DIR, batch)
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).astype("float")
    images = images/255.0 # Normalize images to [0, 1]
    dataset = TensorDataset(torch.FloatTensor(images), torch.LongTensor(labels)) # Convert to torch tensor

    # Splitting the data (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

# Train the autoencoder
def train_autoencoder(autoencoder, optimizer, criterion, train_loader, validation_loader):
    train_loss_history = []
    val_loss_history = []
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

        running_val_loss = 0.0
        for data in validation_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)
            running_val_loss += criterion(outputs, inputs).item()


        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)
        epoch_val_loss = running_val_loss / len(validation_loader)
        val_loss_history.append(epoch_val_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Validation_Loss: {epoch_val_loss:.4f}")

    return train_loss_history, val_loss_history

# Plot the error evolution
def plot_loss(loss_history, title):
    plt.plot(loss_history, label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show()


# Function to visualize original and reconstructed images
def visualize_reconstruction(autoencoder, data_loader, num_images=8):
    autoencoder.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get a batch of test images
        inputs, _ = next(iter(data_loader))
        inputs = inputs.to(device)

        # Pass the images through the autoencoder
        outputs = autoencoder(inputs)

        # Move data to CPU for visualization
        inputs = inputs.cpu().numpy()
        outputs = outputs.cpu().numpy()

        # Plot original and reconstructed images
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            # Original image
            plt.subplot(2, num_images, i + 1)
            original = inputs[i].transpose(1, 2, 0)  # Convert to HWC format
            plt.imshow(original)
            plt.title("Original")
            plt.axis('off')

            # Reconstructed image
            plt.subplot(2, num_images, num_images + i + 1)
            reconstructed = outputs[i].transpose(1, 2, 0)  # Convert to HWC format
            plt.imshow(reconstructed)
            plt.title("Reconstructed")
            plt.axis('off')

        plt.show()