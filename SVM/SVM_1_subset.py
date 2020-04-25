
# Support Vector Machine (SVM)

# build SVM with the training set
# classify test set with the trained SVM

# investigate two different kernels (linear and RBF)
# optimize SVM parameters (C and gamma) by means of cross-validation

# results:
# average accuracy during cross-validation
# for all investigating kernels and all parameters values
# accuracy on the test set with the optimized parameter values

import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import math


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

def import_data_2(path_to_data):
    data = np.genfromtxt(path_to_data, delimiter=",")
    samples = data[:,1:]
    labels = data[:,0]
    return [samples, labels]

if __name__ == '__main__':

    # importing the data sets
    path_to_train_data = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/train_reduced.txt'
    [train_samples, train_labels] = import_data_2(path_to_train_data)
    print("the train data were imported")

    path_to_test_data = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/test_reduced.txt'
    [test_samples, test_labels] = import_data_2(path_to_test_data)
    print("the test data were imported")

    #param_grid = [{'kernel': ['poly'], 'C': [0.001, 1000], 'gamma': [0.1, 0.01], 'degree': [2, 3]}]
    param_grid = {'C': [0.001, 0.1, 1, 10, 1000], 'gamma': [0.1, 0.01, 0.0001, 0.000001, 0.0000001, 0.00000001], 'kernel': ['linear', 'poly', 'rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(train_samples, train_labels)

    predict_labels = grid.predict(test_samples)
    score = accuracy_score(test_labels, predict_labels)

    # results in file SVM_results.md
    with open('SVM_results.md', 'w') as f:
        f.write("# Results for the SVM assignement\n")
        f.write("For a cross-validation done by ```GridSearchCV``` from ```sklearn```, the following accuracies were got:\n\n")
        f.write("```params``` | mean accuracy (```mean_test_score```)\n")
        f.write("--- | ---\n")

        for i in range(len(grid.cv_results_['params'])):
            f.write("{0} | {1}\n".format(grid.cv_results_['params'][i], grid.cv_results_['mean_test_score'][i]))

        f.write('\n')
        f.write("Where the optimized parameter values were:\n\n")
        f.write("```\n")
        f.write("{0}\n".format(grid.best_estimator_))
        f.write("```\n\n")

        f.write("Accuracy on the test set with the optimized parameter values: {0}\n".format(score))
