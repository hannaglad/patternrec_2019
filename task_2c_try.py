import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils as tu
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np

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

# Create list of parameters we want to optimize
lr_list = list(np.arange(0.001, 0.1, 0.001))
batch_size_list = list(range(50,500,10))

#Initialize fixed Parameters
n_class = 10
# Define a maximum number of epochs
max_epochs = 150

# Set variable to compare accuracy over different batch sizes
last_batch_accuracy = 0

#### Load data ####
train_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/train/'
test_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/test/'

# Since pytorch expects a RGB image we have to transform to Grayscale. Then to tensor so it can be used by pytorch
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

# Cycle through batch size untill accuracy converges
for batch_size in batch_size_list:

    train_load = tu.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_load = tu.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Set variables to comapre accuracy for different learning rates
    final_test_accuracy = 0
    final_train_accuracy = 0

    # Cycle through learning rates untill the accuracy converges
    for lr in lr_list:

        # Initialize lists to save results; will eventually be used for the plot once batch size is optimized
        train_accuracy_per_epoch = []
        test_accuracy_per_epoch = []

        # Make the network
        network = PR_CNN()
        criterion = nn.CrossEntropyLoss()
        # Construct an optimizer for weight and bias
        optimizer = optim.SGD(network.parameters(), lr=lr)

        # Train the network
        train_accuracy = 0
        last_train_accuracy = 5

        for epoch in range(max_epochs):
            print("Batch size: {} | LR: {} | Epoch: {}".format(batch_size, lr, epoch))
            train_total = 0
            train_correct = 0
            e = epoch

            if abs(last_train_accuracy-train_accuracy) <= 0.01:
                print("Learning rate: {} | Train accuracy: {:.3f} | Test accuracy: {:.3f}".format(lr, train_accuracy, test_accuracy) + "\n")
                break

            else :
                last_train_accuracy = train_accuracy
                for x, (images, labels) in enumerate(train_load):
                    # Feed the images forward through the network
                    outputs = network(images)
                    loss = criterion(outputs, labels)
                    # Clear gradients
                    optimizer.zero_grad()
                    # Back propagation
                    loss.backward()
                    # Optimize and update parameters
                    optimizer.step()

                    # Calculate the number of correct predictions
                    train_total += labels.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted==labels).sum().item()

                # Calculate training accuracy for the epoch
                train_accuracy = train_correct/train_total

                # Test the network on the test data
                network.eval()
                with torch.no_grad():
                    test_correct=0
                    test_total = 0
                    for images, labels in test_load:
                        # Feed the images to the network
                        outputs = network(images)
                        _, predicted = torch.max(outputs.data,1)
                        test_total += labels.size(0)
                        test_correct += (predicted==labels).sum().item()

                    # Calculate the accuracy on the test data
                    test_accuracy = test_correct/test_total

                # Add to lists of accuracy for later plots
                train_accuracy_per_epoch.append(train_accuracy)
                test_accuracy_per_epoch.append(test_accuracy)


        if (test_accuracy-final_test_accuracy)<=0.01:
            optimized_lr = lr
            batch_accuracy = test_accuracy
            print("Optimized LR for current batch size: {}".format(lr))
            break
        else :
            print("Improvement for LR {:.3f}".format(test_accuracy-final_test_accuracy))
            final_test_accuracy = test_accuracy

    if  (batch_accuracy-last_batch_accuracy)<=0.01:
        print("Optimized batch size : {}".format(batch_size))
        break
    else :
        print("Improvement for batch size {:.3f}".format(batch_accuracy-last_batch_accuracy))
        last_batch_accuracy = batch_accuracy


# Plot the results for best batch size and learning rate
x = list((range(e)))

image_name = "Test_acc.png"
plt.plot(x, test_accuracy_per_epoch)

title = "Test accuracy per epoch at learning rate "+str(lr)+" and batch size" + str(batch_size)
plt.title(title)
plt.xlabel("Learning rate")
plt.ylabel("Test accuracy")
plt.savefig(image_name)
plt.show()

image_name = "Train_acc.png"
title = "Train accuracy per epoch at learning rate "+str(lr)+" and batch size" + str(batch_size)

plt.plot(x, train_accuracy_per_epoch)
plt.title(title)
plt.xlabel("Learning rate")
plt.ylabel("Test accuracy")
plt.savefig(image_name)
plt.show()

# Major source
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
