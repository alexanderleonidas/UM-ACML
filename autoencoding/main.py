import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
import pickle

# Import the dataset
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
# Path to CIFAR-10 folder (adjust as needed)
data_dir = 'cifar-10-batches-py/'  # Replace with your actual folder path

# CIFAR-10 labels
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Helper function to load CIFAR-10 batch files
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).astype("float")
    return images, labels

# Load a single batch for visualization
batch_file = os.path.join(data_dir, 'data_batch_1')  # Change the batch file as needed
images, labels = load_cifar_batch(batch_file)

# Display a grid of images
def visualize_images(images, labels, num_images=16):
    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        img = images[i].transpose(1, 2, 0) / 255.0  # Convert to HWC format and normalize
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.show()

# Visualize the images
visualize_images(images, labels)

# convert
training_set = torch.FloatTensor(images)

# architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=32, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=64)
        pass