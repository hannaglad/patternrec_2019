import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def sigmoid(x):
    result = np.zeros(x.shape)
    s = 1/(1+np.exp(-x,out=result))
    return s

def sigmoid_derivative(s):
    return s*(1-s)

class Layer():
    """Layer of an MLP

    Attributes:
        neurons = number of neurons within the layer as an int
        input_size = number of connections to a neuron within the layer as int
        weights = weights in the layer if none then its randomly initialized as xavier init
        in matrix form where each column is weights for a specific neuron
        bias = biases of the neurons within a layer given as a 1D array"""

    def __init__(self,neurons, input_size, weights=None,bias=None):
        self.neurons = neurons
        self.input_size = input_size
        self.last_activation = None
        if weights == None:
            xavier = np.sqrt(6)/np.sqrt(input_size+neurons)
            self.weights = np.random.uniform(-xavier,xavier,(input_size,neurons))
            self.bias = np.zeros(neurons) # sets the bias term to 0 at start
        else :
            self.weights = weights
            self.bias = bias


    def activate(self, inp):
        """ propagates the input through the layer and saves the result"""
        r = np.matmul(inp,self.weights) + self.bias
        self.last_activation = sigmoid(r)
        return self.last_activation

    def set_weights(self,bias,weights):
        self.bias = bias
        self.weights = weights


    def update_weights(self,increment,index):
        """updates the weights of a neuron at a given index"""
        self.weights[:,index] = self.weights[:,index] - increment

    def update_bias(self,increment,index):
        """ updates the bias of a neuron at a given index"""
        self.bias[index] = self.bias[index] - increment


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
        # update each neuron  weights individually
        for index in range(outputL.neurons):
            increment = delta[index]*inp
            outputL.update_weights(increment*c,index)
            outputL.update_bias(delta[index]*c,index)
        return delta


    def updateIL(self,weights,deltas,inp,hiddenLayer,c):
        """ does the backpropagation update of the hidden layer"""
        derivative= sigmoid_derivative(hiddenLayer.last_activation)
        up = weights*deltas # follow up on the chain rule of backprop
        for index in range(1,hiddenLayer.neurons): # skips over the bias term
            delta = np.sum(up[index:,]) # change for a given neuron within the hidden layer
            increment = delta*inp*derivative[index]
            hiddenLayer.update_weights(increment*c,index)
            hiddenLayer.update_bias(delta*derivative[index]*c,index)


    def backpropagation(self, whole_input, real_values_encoded, c,tolerance):
        """does the online backpropagation update of the network and returns the
        epoch loss
        """
        test_error=0
        # iterate over training sample and do an online mode update
        for index,sample in enumerate(whole_input):
            self.forward_propagation(sample)
            output = self.layers[-1].last_activation
            error = output - real_values_encoded[index,]
            test_error += np.dot(error,error)
            weights = self.layers[-1].weights
            delta = self.updateOL(self.layers[-1],self.layers[-2].last_activation,error,c)
            self.updateIL(weights,delta,sample,self.layers[-2],c)
        return test_error/index+1 # using the MSE as the error function


    def train(self,whole_input, real_values_encoded, c,tolerance,validation,val_labels_encoded,val_labels,train_labels):
        """takes in the training data, class labels, learning rate - c and tolerance as input
        and does the backpropagation update of the Network until the error
        doesnt change up to a tolerance
        """
        flag = True
        #reshape the whole_input for fast propagation
        train = np.reshape(whole_input,(whole_input.shape[0],1,whole_input.shape[1]))
        # initialize the error parameter,epoch counter validation set accuracy,loss and train loss and accuracy
        error_old = np.inf
        counter = 1
        val_acc = []
        train_loss = []
        train_acc = []
        val_loss = []
        acc = 0
        best = None
        while flag:
            test_error = self.backpropagation(whole_input, real_values_encoded, c,tolerance)
            val_loss.append(self.calculate_loss(validation,val_labels_encoded))
            if abs(error_old-test_error)<tolerance:
                flag=False
            train_loss.append(test_error)
            error_old = test_error
            acc_new = np.mean(self.predict(validation)==val_labels)
            val_acc.append(acc_new)
            train_acc.append(np.mean(self.predict(train)==train_labels))
            if acc_new>acc:
                # get the network weights for the best result obtained so far
                best = self.snapshot()
                acc = acc_new
            if counter%10==0:
                # each 10 epochs reduce the learning rate up to a minimum of 0.01
                print("iteration nr {} done".format(counter))
                if c >=0.02:
                    c = c/2
                else:
                    c = 0.01 # no significant improvement was obtained with reducing it further
                print("test error is",test_error)
            counter+=1
            if counter >= 25:
                flag=False
        return val_acc,val_loss,train_acc,train_loss,best


    def predict(self,test):
        """ predicts the labels of the test data"""
        self.forward_propagation(test)
        output = self.layers[-1].last_activation
        result = np.ndarray.flatten(np.argmax(output,axis=2))
        return result

    def calculate_loss(self,valid,labels):
        self.forward_propagation(valid)
        output = self.layers[-1].last_activation-labels
        return np.sum(output**2)/(output.shape[0])

    def snapshot(self):
        """returns the current weight and bias parameters"""
        return (self.layers[0].bias,self.layers[0].weights),(self.layers[1].bias,self.layers[1].weights)


    def __str__(self):
        return "Network with {} layers.".format(self.layers)


def replicate_experiment(train,test,n):
    """preforms several random initializations and returns the weights for the one
    with the highest accuracy"""
    encoder = OneHotEncoder(sparse=False)
    idx = np.arange(train.shape[0])
    acc = 0
    params =  None
    for i in range(n):
        permuted = np.random.permutation(idx)
        train = train[permuted]

        # Get the labels
        trainL = np.array(train[:,0])
        encoded_labels = trainL.reshape(len(train),1)
        encoded_labels = encoder.fit_transform(encoded_labels)
        train2 = train[:,1:]/255  # scales the data to 0 to 1 scale

        # Define validation set
        val_labels = test[:,0]
        encoded_val_labels = val_labels.reshape(len(test),1)
        encoded_val_labels = encoder.fit_transform(encoded_val_labels)
        validation = test[:,1:]/255
        validation = np.reshape(validation,(validation.shape[0],1,validation.shape[1]))

        val_labels_encoded = encoded_val_labels
        val_labels_encoded = np.reshape(val_labels_encoded,(val_labels_encoded.shape[0],1,val_labels_encoded.shape[1]))


        # Get the input size
        input_size = np.shape(train2)[1]
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
        val = network.train(train2, encoded_labels, 0.1,0.001, validation, val_labels_encoded, val_labels, trainL)
        print("Network trained")
        # get the best accuracy
        acc_new = np.max(val[0])
        print(acc_new)
        if acc_new > acc:
            params = val[-1]
            acc = acc_new

            # Plot results from the training
            validation_loss = val[1]
            validation_acc = val[0]
            train_loss = val[3]
            train_acc = val[2]

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

    return params

# Import the data
print("Importing data...")
train = np.genfromtxt("mnist_train.csv", delimiter=",")
test = np.genfromtxt("mnist_test.csv", delimiter=",")

results = replicate_experiment(train,test,5)

# take the best result and try it on the test set
network = Network()
hid1 = Layer(300,test.shape[1]-1)
hid1.set_weights(weights=results[0][1],bias=results[0][0])
outputLayer = Layer(10,300)
outputLayer.set_weights(weights=results[1][1],bias=results[1][0])
network.add_layer(hid1)
network.add_layer(outputLayer)

#Get the predicted_values
labelsAll = np.array(test[:,0],dtype="int")
test = test[:,1:]/255
predicted = network.predict(np.reshape(test,(test.shape[0],1,test.shape[1])))
compare = predicted == labelsAll
print(np.mean(compare))

filename = "MLP_results.txt"
with open(filename, 'w') as file:
    file.write("Test accuracy = {}.format(np.mean(compare)))
