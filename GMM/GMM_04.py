
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plot
from scipy.optimize import linear_sum_assignment
from collections import Counter

# Task 5
# Classify the molecules of the validation set using KNN
# Distance: approximate Graph Edit Distance (GED)

def print_cost_matrix(cost_matrix, g1, g2):
    """
    Function that nicely (more or less) print our cost_matrix
    """
    print('    ', end= '')
    for node2, label2 in g2.nodes(data='label'):
        print(label2, end='   ')
    print()
    i = 0
    for node1, label1 in g1.nodes(data='label'):
        print(label1, end=' ')
        print(cost_matrix[i])
        i += 1
    while i < len(cost_matrix):
        print(end= '  ')
        print(cost_matrix[i])
        i += 1


def build_cost_matrix(g1, g2, Cn, Ce):

    ####### slides 17-20, Lecture 10 #######

    nn1 = g1.number_of_nodes()
    nn2 = g2.number_of_nodes()

    n_m = nn1 + nn2
    # empty matrix of size (n+m) x (n+m)
    cost_matrix = np.zeros((n_m, n_m))

    i = 0
    for node1, label1 in g1.nodes(data = 'label'):
        j = 0
        for node2, label2 in g2.nodes(data = 'label'):
            if label1 != label2:
                cost = 2*Cn+Ce*abs(g2.degree(node2)-g1.degree(node1))
                cost_matrix[i,j] = cost
            j += 1
        i += 1
    cost_matrix[i:,:j]=np.inf
    cost_matrix[:i,j:]=np.inf
    z=0
    for node1, label1 in g1.nodes(data = 'label'):
       cost_matrix[z,j]=Cn+Ce*g1.degree(node1)
       j+=1
       z+=1
    z=0
    for node2, label2 in g2.nodes(data = 'label'):
        cost_matrix[i,z]=Cn+Ce*g2.degree(node2)
        i+=1
        z+=1
    return cost_matrix

if __name__ == "__main__":

    # restore from files
    f = open('train.var', 'rb')
    data_train = pickle.load(f)
    f.close()

    f = open('valid.var', 'rb')
    data_valid = pickle.load(f)
    f.close()

    grid_list = []

    Cn = np.arange(0.5, 5, 0.5)
    Ce = np.arange(0.5, 5, 0.5)

    for cn in Cn:
        for ce in Ce :

            results = np.zeros((len(data_valid),len(data_train)))
            labels = [] # create the validation label set
            labels_train = [data_train[j].graph['acitivity'] for j in range(len(data_train))] # create the train label set
            # iterate over the validation set and the train set and calculate the GED
            for i in range(len(data_valid)):
                g1 = data_valid[i]
                label1 = g1.graph['acitivity'] # was lazy to change to saskias new version so left acitivity instead of activity ^^
                labels.append(label1)
                for j in range(len(data_train)):
                    g2 = data_train[j]
                    cost_matrix = build_cost_matrix(g1, g2, cn, ce)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    ged = cost_matrix[row_ind, col_ind].sum()
                    results[i,j]=ged
            acc = []
            for k in range(21):
                # find the k graphs with min distance
                idx = np.argpartition(results,k,axis=1)
                guessed_labels= []
                labels = np.array(labels)
                for i in range(len(data_valid)):
                    #get the labels of the closest graphs
                    nn = labels_train[idx[i,k]]
                    # find the most common label
                    vote = Counter(nn).most_common(1)
                    # if vote is a tuple it means only one classifier is the best
                    if type(vote) == tuple:
                        clasifier = vote[0]
                        # if ties occur takes a random of the best classifiers
                    else:
                        clasifier = vote[0][0]
                    guessed_labels.append(clasifier)
                truth = labels==np.array(guessed_labels)
                acc.append(np.mean(truth))
                print("k:",k)
            print("ce:", ce)
            grid_list.append([cn, ce, max(acc), acc.index(max(acc))])

        print("cn:", cn)

    print(grid_list)
    max_accuracy = grid_list.index(max(grid_list, key=lambda x:x[2]))
    max_entry = grid_list[max_accuracy]
    with open("GMM_results.txt", 'w') as file:
        for item in grid_list:
            file.write("Cn: "+str(item[0])+" Ce: "+str(item[1])+" Accuracy: "+str(item[2]))
        file.write("Best Accuracy {} obtained with Cn:{}, Ce:{} and K:{}".format(max_entry[2], max_entry[0], max_entry[1]))
