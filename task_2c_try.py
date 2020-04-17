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

# Create list of parameters we want to test
lr_list = list(np.arange(0.001, 0.1, 0.001))
batch_size = 50

#Initialize fixed Parameters
n_class = 10
# Define a maximum number of epochs
max_epochs = 150

# Initialize list to store results
train_acc_list = []
test_acc_list = []

# Set variable to compare accuracy over different learning rates
final_test_accuracy = 0

#### Load data ####
train_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/train/'
test_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/test/'

# Since pytorch expects a RGB image we have to transform to Grayscale. Then to tensor so it can be used by pytorch
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

train_data = datasets.ImageFolder(root=train_path, transform=transform)
train_load = tu.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.ImageFolder(root=test_path, transform=transform)
test_load = tu.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Cycle through learning rates untill the accuracy converges
for lr in lr_list:
    train_last_correct = 0
    test_last_correct = 0

    # Make the network
    network = PR_CNN()
    criterion = nn.CrossEntropyLoss()
    # Construct an optimizer for weight and bias
    optimizer = optim.SGD(network.parameters(), lr=lr)

    # Train the network

    train_accuracy = 0
    last_train_accuracy = 5

    for epoch in range(max_epochs):
        running_train_total = 0
        running_train_correct = 0
        e = epoch

        if abs(last_train_accuracy-train_accuracy) < 0.01:
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

                # Calculate the number of correct predictions for the batch
                total = labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted==labels).sum().item()
                step_train_accuracy = correct/total

                # Add to the number of correct predictions for the epoch
                running_train_total += total
                running_train_correct += correct

                if (x+1)%100 == 0:
                    print("Epoch {} | Iteration {} |".format(e+1, x+1))

            train_accuracy = running_train_correct/running_train_total

            # Test the network on the test data
            network.eval()
            with torch.no_grad():
                correct=0
                total = 0
                for images, labels in test_load:
                    # Feed the images to the network
                    outputs = network(images)
                    _, predicted = torch.max(outputs.data,1)
                    total += labels.size(0)
                    correct += (predicted==labels).sum().item()

                # Calculate the accuracy on the test data
                test_accuracy = correct/total

    print("Learning rate: {} | Convergence in {} epochs | Train accuracy: {} | Test accuracy: {}".format(lr, e, train_accuracy, test_accuracy) + "\n")
    train_acc_list.append(train_accuracy)
    test_acc_list.append(test_accuracy)

    if (test_accuracy-final_test_accuracy)<=0.01:
        optimized_lr = lr
        print("Optimized lr : {}".format(lr))
        break
    else :
        print("Improvement {}".format(test_accuracy-final_test_accuracy))
        final_test_accuracy = test_accuracy


# Plot the results for best learning rate

### Plot the resultsnd epochs
# Find the best learning rate and plot the improvement of accuracy per epoch for that learning rate

image_name = "Test_acc_learning_Rate.png"

index = lr_list.index(optimized_lr)
x = lr_list[:index+1]

plt.plot(x, test_acc_list)
plt.set_title("Test accuracy for different learning rates")
plt.set_xlabel("Learning rate")
plt.set_ylabel("Test accuracy")
plt.savefig(image_name)
plt.show()

image_name = "Train_acc_learning_Rate.png"
plt.plot(x, train_acc_list)
plt.set_title("Train accuracy for different learning rates")
plt.set_xlabel("Learning rate")
plt.set_ylabel("Test accuracy")
plt.savefig(image_name)
plt.show()


# Save the model to use for batch size optimization
path = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/network/'
filename = "network.pt"
state = {
'state_dict': network.state_dict(),
'optimizer': optimizer.state_dict(),
}
torch.save(state, path+filename)

# Major source
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
