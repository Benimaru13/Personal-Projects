import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pathlib import Path
from PIL import Image


# Initialize CNN Model
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images) but for mine, it is also 1.
            num_classes: int
                The number of classes we want to predict, in our case 3 (HAMS_C, HAMS_K, HAMS_Z).
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Third convolutional layer: 16 input channels, 32 output channels, 3x3 kernel, stride 1, padding 1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Fully connected layer: 32*16*16 input features (after three 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)
        # add a dropout rate to prevent overfitting and co-adaptation
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv3(x))  # Apply third convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.dropout(x)        # Apply dropout
        x = self.fc1(x)            # Apply fully connected layer
        return x

def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  # Set the model back to training mode


device = "cuda" if torch.cuda.is_available() else "cpu"

'''
ImageFolder takes in a directory containing subdirectories, where each subdirectory corresponds to a class and contains images belonging to that class.
The transform argument allows you to specify the transformations to apply to the images when they are loaded.
 In this case, we are converting the images to tensors and normalizing them with a mean of 0.5 and a standard deviation of 0.5.
'''
# Define the transformations to apply to the images
# The transforms.Compose() function allows you to chain multiple transformations together. In this case, we are converting the images to grayscale, then to tensors, and finally normalizing them.
# add various transormers for varioation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # forcing it to be grayscale, which means it will have only one channel. This is done to ensure that all images are in the same format and to reduce the computational complexity of the model.
    transforms.RandomRotation(10), # Randomly rotate the images by up to 10 degrees. This is a data augmentation technique that helps the model generalize better by introducing some variation in the training data.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Randomly translate the images by up to 10% in both the x and y directions. This is another data augmentation technique that helps the model generalize better by introducing some variation in the training data.
    transforms.ToTensor(), # Converting the images to tensors, which is the format that PyTorch models expect. This transformation also scales the pixel values to be between 0 and 1.
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalizing the pixel values to be between -1 and 1. The mean and std values are set to 0.5 to achieve this normalization.
])
# Specify the directory containing the processed writer datasets
WRITERS_DIR = Path("C:\\Users\\BC-Tech\\Documents\\Chibueze's Code\\Personal-Projects\\Handwriting-Distinguisher\\data\\processed")

# Attach the transform to the dataset
dataset = ImageFolder(WRITERS_DIR, transform=transform)

# Until this point, we have created a dataset object that can be used to load and preprocess the images. Now, we will create a DataLoader to load the data in batches and shuffle it for training.
# You don't need to save the tensors anywhere
# It only applies the loader when it is run and dones\t permanently modify it
# batchsize 16 only iterates through 16 random items in the dataset

"""
loader = DataLoader(dataset, batch_size=16, shuffle=True)
images, labels = next(iter(loader)) # labels look like {'HAMS_C': 0, 'HAMS_K': 1, 'HAMS_Z': 2}

print(f"Batch of images shape: {images.shape}")
"""

# Spliting dataset into train/ validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# epoch is one full pass through the training dataset
model = CNN(in_channels=1, num_classes=3) # Initialize the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # weight decay is a regularization technique that adds a penalty to the loss function based on the magnitude of the model's weights. This helps prevent overfitting by discouraging the model from relying too heavily on any particular feature or set of features.  

num_epochs = 25
min_loss = float('inf')
smallest_loss_epoch = 0
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for images, labels in train_loader:
        # Move data and targets to the device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: compute the model output
        scores = model(images)
        loss = criterion(scores, labels)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()
    if loss.item() < min_loss or min_loss == float('inf'):
        min_loss = loss.item()
        smallest_loss_epoch = epoch
    print(f"Loss: {loss.item():.4f}")
    # Final accuracy check on training and test sets
    print("Training Accuracy:")
    check_accuracy(train_loader, model)
    print("Validation Accuracy:")
    check_accuracy(val_loader, model)
print(f"Smallest loss of {min_loss:.4f} occurred at epoch {smallest_loss_epoch + 1}")
 

        
        
