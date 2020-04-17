
# Support Vector Machine (SVM)

# build SVM with the training set
# classify test set with the trained SVM

# todo investigate two different kernels (linear, RBF,...)
# todo optimize SVM parameters (C and gamma) by means of cross-validation

# results:
# average accuracy during cross-validation for all investigating kernels
# and all parameters values
# accuracy on the test set with the optimized parameter values

import csv
import numpy as np
from sklearn.svm import SVC

def import_data(path_to_data):
    """
    :param path_to_data
    :return: a list containing the samples (0) and the labels (1)
    """
    with open(path_to_data, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        matrix = np.array(data, dtype = int)
        samples = matrix[:, 1:]
        labels = matrix[:, 0]
    return [samples, labels]

if __name__ == '__main__':

    # importing the data sets
    path_to_train_data = '../mnist-csv-format/mnist_train.csv'
    [train_samples, train_labels] = import_data(path_to_train_data)
    print("the train data were imported")

    path_to_test_data = '../mnist-csv-format/mnist_test.csv'
    [test_samples, test_labels] = import_data(path_to_test_data)
    print("the test data were imported")

    # from here, my computer (saskia) is a little too weak to compute...

    # train a SVM with the training set
    # linear kernel
    linear_svm = SVC (kernel ='linear') # SVC stands for Support Vector Classifier
    linear_svm.fit(train_samples, train_labels)

    # classify test set with the trained SVM
    predicted_labels = linear_svm.predict(test_samples)



