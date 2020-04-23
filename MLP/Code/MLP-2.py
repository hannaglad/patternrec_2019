import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def sigmoid(x):
    s = 1/(1+np.exp(-x,dtype="float32"))
    return s

def sigmoid_derivative(s):
    return s*(1-s)

class Layer():
    """Layer of an MLP

    Attributes:
        neurons = number of neurons within the layer as an int
        input_size = number of connections to a neuron within the layer as int
        weights = weights are randomly initialized as xavier init
        in matrix form where each column is weights for a specific neuron
        bias = biases of the neurons within a layer given as a 1D array, randomly initialized as 0"""

    def __init__(self,neurons, input_size,bias=None):
        self.neurons = neurons
        self.input_size = input_size
        self.last_activation = None
        xavier = np.sqrt(6)/np.sqrt(input_size+neurons)
        self.weights = np.random.uniform(-xavier,xavier,(input_size,neurons)).astype("float32")
        self.bias = np.zeros(neurons,dtype="float32") # sets the bias term to 0 at start


    def activate(self, inp):
        """ propagates the input through the layer and saves the result"""
        r = np.matmul(inp,self.weights) + self.bias
        self.last_activation = sigmoid(r)
        return self.last_activation

    def set_weights(self,bias,weights):
        """set the bias and weights of a network manually"""
        self.bias = bias
        self.weights = weights


    def update_weights(self,increment):
        """updates the weights of a neuron at a given index"""
        self.weights = self.weights - increment


    def update_bias(self,increment):
        """ updates the bias of a neuron at a given index"""
        self.bias = self.bias - increment


    def __str__(self):
        return "{} {} {} {} {} {} {}".format("Layer with", self.neurons, "perceptrons.", "Input size is", self.input_size, "Weight shape is", np.shape(self.weights))


class Network():
    """Entire mlp network
    initialized as an empty list of layers, layers should be added via add_layer method"""


    def __init__(self):
        self.layers = []


    def add_layer(self, layer):
        self.layers.append(layer)


    def forward_propagation(self, single_input):
        """propagates the sample through the entire network"""
        for layer in self.layers:
            single_input = layer.activate(single_input)


    def updateOL(self,outputL,inp,error,c):
        """ does the back propagation update of the outer layer and returns the
        error times derivative term"""
        derivative = sigmoid_derivative(outputL.last_activation)
        delta = derivative*error
        # update the bias and weights for each neuron in the layer
        outputL.update_bias(delta*c)
        increment = delta*inp.T
        outputL.update_weights(increment*c)
        return delta


    def updateIL(self,weights,deltas,inp,hiddenLayer,c):
        """ does the backpropagation update of the hidden layer"""
        derivative= sigmoid_derivative(hiddenLayer.last_activation)
        up = weights*deltas # follow up on the chain rule of backprop
        delta = np.sum(up,axis=1) # change for neurons within the hidden layer
        # update the bias and weight for each neuron in the layer
        hiddenLayer.update_bias(delta*derivative*c)
        increment = delta*inp.T*derivative
        hiddenLayer.update_weights(increment*c)


    def backpropagation(self, whole_input, real_values_encoded, c):
        """does the online backpropagation update of the network with learning rate c
        """
        # iterate over training sample and do an online mode update
        for index,sample in enumerate(whole_input):
            self.forward_propagation(sample)
            output = self.layers[-1].last_activation
            error = output - real_values_encoded[index,]
            # get the nonupdated weights for the backprop through the hiddenlayer
            weights = self.layers[-1].weights
            delta = self.updateOL(self.layers[-1],self.layers[-2].last_activation,error,c)
            self.updateIL(weights,delta,sample,self.layers[-2],c)


    def train(self,trainset,train_labels,validation,val_labels,c,tolerance,maxiter):
        """takes in the training data, class labels, learning rate - c and tolerance as input
        and does the backpropagation update of the Network until the error
        doesnt change up to a tolerance
        """
        encoder = OneHotEncoder(sparse=False)
        # encode the labels for the train set and validation set
        train_values_encoded = train_labels.reshape(len(train),1)
        train_values_encoded = encoder.fit_transform(train_values_encoded)
        val_labels_encoded  = val_labels.reshape(len(validation),1)
        val_labels_encoded  = encoder.fit_transform(val_labels_encoded )
        # reshape the validation labels and train labels so that we can do the entire set at once
        val_labels_encoded = np.reshape(val_labels_encoded,(val_labels_encoded.shape[0],1,val_labels_encoded.shape[1]))
        train_values_encoded = np.reshape(train_values_encoded,(train_values_encoded.shape[0],1,train_values_encoded.shape[1]))
        # initialize the error parameter,epoch counter validation set accuracy,loss and train loss and accuracy
        error_old = np.inf
        counter = 1
        val_acc, train_loss,train_acc, val_loss = [],[],[],[]
        acc = 0
        best = None
        for epoch in range(maxiter+1):
            self.backpropagation(trainset, train_values_encoded, c)
            acc_new,v_loss = self.calculate_loss_and_acc(validation,val_labels_encoded,val_labels)
            val_loss.append(v_loss)
            val_acc.append(acc_new)
            if acc_new>acc:
                # get the network weights for the best result obtained so far
                best = self.snapshot()
                acc = acc_new
            t_acc,t_error = self.calculate_loss_and_acc(trainset,train_values_encoded,train_labels)
            train_loss.append(t_error)
            train_acc.append(t_acc)
            # if the error on the train set drops bellow a tolerance finish the process
            if abs(error_old-t_error)<tolerance:
                break
            # else update the error to current error
            error_old = t_error
            # each 10 epochs drop the learnrate by half
            if counter%10==0:
                # each 10 epochs reduce the learning rate
                print("iteration nr {} done".format(counter))
                c = c/2
                print("test error is",t_error)
            counter+=1
        return val_acc,val_loss,train_acc,train_loss,best


    def predict(self,test):
        """ predicts the labels of the test data"""
        self.forward_propagation(test)
        output = self.layers[-1].last_activation
        print(output.shape)
        result = np.ndarray.flatten(np.argmax(output,axis=2))
        return result


    def calculate_loss_and_acc(self,valid,labels_encoded,labels):
        """ caluclates the loss and accuracy of the presented dataset"""
        self.forward_propagation(valid)
        output = self.layers[-1].last_activation
        # get the neuron with the highest activation value
        result = np.ndarray.flatten(np.argmax(output,axis=2))
        # compare to given labels and also calculate the loss based on MSE
        return np.mean(result==labels),np.sum((labels_encoded - self.layers[-1].last_activation)**2)/(output.shape[0])

    def snapshot(self):
        """returns the current weight and bias parameters"""
        return (self.layers[0].bias,self.layers[0].weights),(self.layers[1].bias,self.layers[1].weights)


    def __str__(self):
        return "Network with {} layers.".format(self.layers)


def replicate_experiment(train,test,n,lr,tresh,miter):
    """preforms n random initializations of the network with learnrate lr,
    error treshold tresh and maximumepoch number miter,
    plots the acuracy and loss and returns the weights for the one
    with the highest accuracy on the validation set"""
    idx = np.arange(train.shape[0])
    acc = 0
    params =  None
    trainset,trainL =train[:,1:], train[:,0]
    valset,valL = test[:,1:],test[:,0]
    # reshape the validationset and trainset so that they can be propagated all at once
    valset = np.reshape(valset,(valset.shape[0],1,valset.shape[1]))
    trainset = np.reshape(trainset,(trainset.shape[0],1,trainset.shape[1]))
    input_size = trainset.shape[2]
    print(input_size)
    for i in range(n):
        # each time randomly shuffle the train data
        permuted = np.random.permutation(idx)
        trainset = trainset[permuted]
        trainL = trainL[permuted]
        # Get the input size
        print("Data imported")
        print("Creating network...")
        network = Network()
        hid1 = Layer(300,input_size)
        outputLayer = Layer(10,300)
        network.add_layer(hid1)
        network.add_layer(outputLayer)
        print("Layers added")
        print("training...")
        # Train the network
        val = network.train(trainset, trainL,valset,valL,lr,tresh,miter)
        print("Network trained")

        # Plot results from the training
        validation_loss,validation_acc, train_loss,train_acc = val[1],val[0],val[3],val[2]
        title = "MLP_test_it_"+str(i)+".png"
        fig, ax = plt.subplots()
        x = list(range(len(val[2])))
        ax.plot(x, train_acc, color="green", label="Accuracy")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Training accuracy")

        ax2 = ax.twinx()
        ax2.plot(x, train_loss, color="red", label="Loss")
        ax2.set_ylabel("Training loss")
        fig.savefig(title)

        # Plot results from the validation
        title = "MLP_training_it_"+str(i)+".png"
        fig, ax = plt.subplots()

        ax.plot(x, validation_acc, color="green", label="Accuracy")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Validation accuracy")

        ax2 = ax.twinx()
        ax2.plot(x, validation_loss, color="red", label="Loss")
        ax2.set_ylabel("Validation loss")
        fig.savefig(title)
        # get the best accuracy of the iteration
        acc_new = np.max(val[0])
        print(acc_new)
        # if this is the best result so far take the iteration params as the best ones
        if acc_new > acc:
            params = val[-1]
            acc = acc_new
    return params

# Import the data
print("Importing data...")
train = np.genfromtxt("mnist_train.csv", delimiter=",",dtype="float32")
val = np.genfromtxt("val_csv.csv", delimiter=",", dtype="float32")
test = np.genfromtxt("mnist_test.csv", delimiter=",",dtype="float32")
results = replicate_experiment(train,val,1,0.1,0.001,40)

# take the best result and try it on the test set
network = Network()
hid1 = Layer(300,test.shape[1]-1)
hid1.set_weights(weights=results[0][1],bias=results[0][0])
outputLayer = Layer(10,300)
outputLayer.set_weights(weights=results[1][1],bias=results[1][0])
network.add_layer(hid1)
network.add_layer(outputLayer)

#Get the predicted_values
labelsAll = np.array(test[:,0])
test = test[:,1:]
predicted = network.predict(np.reshape(test,(test.shape[0],1,test.shape[1])))
compare = predicted == labelsAll
print(np.mean(compare))


filename = "MLP_results.txt"
with open(filename, 'w') as file:
    file.write("Test accuracy = ", np.mean(compare))
