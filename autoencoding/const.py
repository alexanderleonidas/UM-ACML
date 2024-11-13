import torch

EPOCHS = 10

# Path to CIFAR-10 folder (adjust as needed)
DATA_DIR = 'cifar-10-batches-py/'  # Replace with your actual folder path

# CIFAR-10 labels
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set the device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')