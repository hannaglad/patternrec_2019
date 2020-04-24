import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils as tu
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
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

# Create list of parameters we want to optimize
lr_list = list(np.arange(0.001, 0.1, 0.001))

batch_reduction = 25
batch_size_list = list(reversed(range(25,100,batch_reduction)))

#Initialize fixed Parameters
n_class = 10
max_epochs = 150

#### Load data ####
train_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/train/'
test_path = '~/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/val/'

# Since pytorch expects a RGB image we have to transform to Grayscale. Then to tensor so it can be used by pytorch
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

# Perform 5 random intializations
for it in range(1):

    # Set variable to compare accuracy over different batch sizes
    last_batch_accuracy = 0
    batch_accuracy_list = []

    # Cycle through batch size untill accuracy converges
    for batch_size in batch_size_list:

        train_load = tu.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_load = tu.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        # Set variables to comapre accuracy for different learning rates within the batch size
        last_test_accuracy = 0

        # Cycle through learning rates untill the accuracy converges
        for lr in lr_list:

            # Initialize lists to save results; will eventually be used for the plot once batch size is optimized
            train_accuracy_per_epoch = []
            test_accuracy_per_epoch = []
            last_train_accuracy_list = []
            last_test_accuracy_list = []

            train_loss_per_epoch = []
            test_loss_per_epoch =  []

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
                epoch_train_loss = []
                epoch_test_loss = []

                # Untill the accuracy starts stabilising for the training set, keep going through epochs
                if abs(last_train_accuracy-train_accuracy) <= 0.01:
                    print("Learning rate: {} | Train accuracy: {:.3f} | Test accuracy: {:.3f}".format(lr, train_accuracy, test_accuracy))
                    break

                else :
                    last_train_accuracy = train_accuracy

                    last_train_accuracy_list = train_accuracy_per_epoch
                    last_test_accuracy_list = test_accuracy_per_epoch

                    for x, (images, labels) in enumerate(train_load):
                        # Feed the images forward through the network
                        outputs = network(images)
                        loss = criterion(outputs, labels)

                        epoch_train_loss.append(loss.item())
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
                    train_accuracy_per_epoch.append(train_accuracy)

                    # Calcualte training loss for the epoch
                    mean_train_loss = mean(epoch_train_loss)
                    train_loss_per_epoch.append(mean_train_loss)

                    # Test the network on the test data
                    network.eval()
                    with torch.no_grad():
                        test_correct=0
                        test_total = 0
                        for images, labels in test_load:
                            # Feed the images to the network
                            outputs = network(images)
                            loss = criterion(outputs, labels)
                            epoch_test_loss.append(loss.item())
                            _, predicted = torch.max(outputs.data,1)
                            test_total += labels.size(0)
                            test_correct += (predicted==labels).sum().item()

                        # Calculate the accuracy on the test data for the epoch
                        test_accuracy = test_correct/test_total
                        test_accuracy_per_epoch.append(test_accuracy)

                        # Calculate the loss on the test data for the epoch
                        mean_test_loss = mean(epoch_test_loss)
                        test_loss_per_epoch.append(mean_test_loss)

            print("Improvement for LR {:.3f}".format(test_accuracy-last_test_accuracy)+"\n")

            if abs(test_accuracy-last_test_accuracy)<=0.01:
                optimized_lr = lr-0.001
                batch_accuracy = last_test_accuracy
                print("Optimized LR for current batch size: {}".format(optimized_lr))
                break
            else :
                last_test_accuracy = test_accuracy

        print("Improvement for batch size {:.3f}".format(batch_accuracy-last_batch_accuracy)+"\n")

        if abs(batch_accuracy-last_batch_accuracy)<=0.01:
            optimized_batch_size = batch_size+batch_reduction
            flag = True
            if abs(batch_accuracy-last_batch_accuracy) >=0.95:
                break
                print("Optimized batch size : {}".format(optimized_batch_size))

        else :
            batch_accuracy_list.append(batch_accuracy)
            last_batch_accuracy = batch_accuracy

    if flag == False :
        optimized_batch_size = batch_size_list[batch_accuracy_list.index(max(batch_accuracy_list))]

    print("Optimized batch size : {}".format(optimized_batch_size))
    filename = "CNN_results.txt"
    with open(filename, 'a') as file:
        file.write("Iteration {} \nOptimized batch size {} \nOptimized learning rate {} \nFinal training accuracy {} \nFinal test accuracy {}\n\n".format(it, optimized_batch_size, optimized_lr, last_train_accuracy, last_batch_accuracy))

    # Plot the results for best batch size and learning rate
    x = list((range(e)))

    image_name = "CNN_Validation_normal_it_"+str(it)+".png"
    title = "Validation Loss and Accuracy at learning rate "+str(optimized_lr)+" and batch size" + str(optimized_batch_size)

    fig, ax = plt.subplots()
    ax.plot(x, last_test_accuracy_list, color="green")
    ax.set_title(title)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation accuracy")

    ax2 = ax.twinx()
    ax2.plot(x, test_loss_per_epoch, color="red")
    ax2.set_ylabel("Validation loss")
    plt.show()
    fig.savefig(image_name)

    image_name = "CNN_Train_normal_it_"+str(it)+".png"
    title = "Training loss and accuracy "+str(optimized_lr)+" and batch size" + str(optimized_batch_size)
    fig, ax = plt.subplots()
    ax.plot(x, last_train_accuracy_list, color="green")
    ax.set_title(title)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Training accuracy")

    ax2 = ax.twinx()
    ax2.plot(x, train_loss_per_epoch, color="red")
    ax2.set_ylabel("Training loss")
    fig.savefig(image_name)

    # Save the model for use on test set
    model_save_path = "/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/mnist/network/final_network_normal"+str(it)+".pt"
    torch.save(network.state_dict(), model_save_path)

# Major source
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
