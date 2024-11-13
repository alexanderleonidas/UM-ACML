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
    dataset = TensorDataset(torch.FloatTensor(images), torch.LongTensor(labels))

    # Splitting the data (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

# Step 4: Plot the error evolution
def plot_loss(loss_history):
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Autoencoder Training Loss')
    plt.show()