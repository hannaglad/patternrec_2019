import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils as tu
import torch.optim as optim
import torch
import numpy as np
from statistics import mean

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28, 28)

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels ands the kernel size
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=3),
            nn.LeakyReLU()
        )

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=96, kernel_size=3, stride=1),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(1536, 10)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


batch_size = 50
network = PR_CNN()
criterion = nn.CrossEntropyLoss()

model_save_path = "/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/network/final_network_normal0.pt"
network.load_state_dict(torch.load(model_save_path))
network.eval()

transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
test_path = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/test/'
test_data = datasets.ImageFolder(root=test_path, transform=transform)
test_load = tu.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

with torch.no_grad():
    test_correct=0
    test_total = 0
    for images, labels in test_load:
        #Feed the images to the network
        outputs = network(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data,1)
        test_total += labels.size(0)
        test_correct += (predicted==labels).sum().item()

    # Calculate the accuracy on the test data
    test_accuracy = test_correct/test_total

filename = "CNN_normal_test_results.txt"
with open(filename, 'w') as file:
    file.write("Accuracy on the test data" + str(test_accuracy))
