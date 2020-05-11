# DTW

from fastdtw import fastdtw
from scipy.spatial import distance
import numpy as np
import os
import sys
from multiprocessing import Process, Pool
from functools import partial
from sklearn.metrics import auc
import matplotlib.pyplot as plt

euclidean = distance.euclidean
##### Functions for recall / precison
def divide_dict(distance_dict, threshold):
    """
    :param distance_dict:
    :param threshold:
    :return: a list of 2 dictionaries:
     one containing the elements considered as positive with our threshold
     the other the elements considered as negative
    """

    positive_dict = {}
    negative_dict = {}

    for k, v in distance_dict.items():
        if v < threshold:
            positive_dict[k] = v
        else:
            negative_dict[k] = v

    return [positive_dict, negative_dict]

def find_n_equidistant(n, min, max):
    return [min + x * ((max-min)/(n+1)) for x in range(1, n+1)]

def get_tp_fp(word, pos_dict, transcription_dict):
    """
    Function that calculate the true positive and false positive from the pos_dict.
    :param pos_dict:
    :return:
    """
    tp = 0
    fp = 0

    for key in pos_dict.keys() :
        if transcription_dict[key].lower() in word.lower() or word.lower() in transcription_dict[key].lower() :
            tp += 1
        else :
            fp += 1

    return [tp, fp]

def precision(tp, fp):
    if tp == 0 :
        return 0
    else :
        return tp / (tp+fp)

def recall(tp, fn):
    if tp == 0 :
        return 0
    else :
        return tp / (tp + fn)

def get_fn(word, neg_dict, transcription_dict):
    fn = 0

    for key in neg_dict.keys() :
        if transcription_dict[key].lower() in word.lower() or word.lower() in transcription_dict[key].lower() :
            fn += 1
    return fn
#############################
############### IMPORT DATA
# Load all of the feature vectors
root = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/featuresdah/'
folders1 = list(range(270,280,1))
folders2 = list(range(300,305,1))
folders = folders1+folders2

giant_feature_dict = {}
for f in folders:
    path = root+str(f)+'/'

    for file in os.listdir(path):
        filename = str(file).strip('.png.txt')
        features = np.genfromtxt(path+file, delimiter=",")
        giant_feature_dict[filename] = features

# Load the keywords to search for
keyword_file = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/PatRec17_KWS_Data-master/word_locations/keyword_locations.txt'

keyword_dict = {}
with open(keyword_file, 'r') as file:
    for line in file:
        content = line.split(' ')
        key = content[0].strip()
        val = content[1].strip()
        keyword_dict[key] = val

# Load the transcription of all words
transcription_file = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/PatRec17_KWS_Data-master/word_locations/stripped_transcription.txt'
transcription_dict = {}
with open(transcription_file, 'r') as file:
    for line in file:
        content = line.split(' ')
        key = content[0].strip()
        val = content[1].strip()
        transcription_dict[key] = val

### ###################  DTW ###
train = list(range(270,280,1))
val = list(range(300, 305, 1))

i = 0
correct = 0
false = 0
result_dict = {}

## train
# define valid keys if only want to examine part of the data
validKeys = []
for key in keyword_dict.keys() :
    if any(str(train_image) in key for train_image in train):
        validKeys.append(key)

#transcription_dict_ref = {key:value for key,value in transcription_dict.items() if key in validKeys}
total = len(keyword_dict.keys())
n_thresholds = 150
average_recall = np.ndarray((n_thresholds,len(keyword_dict.keys())))
average_precision = np.ndarray((n_thresholds,len(keyword_dict.keys())))

for key in keyword_dict.keys():


    # Set query
    query = giant_feature_dict[key]
    # Get reference keys
    reference = [k for k,value in transcription_dict.items() if k != key]
    reference_features = [giant_feature_dict[ref] for ref in reference]

    # DTW
    part = partial(fastdtw, query, dist=euclidean)
    process = Pool(processes=200)
    all_distances = process.map(part, reference_features)
    all_distances = np.array(all_distances)[:,0].tolist()
    process.close()
    process.join()

    # Make dictionnary of all distances
    distance_dict = {reference[i]:all_distances[i] for i in range(len(all_distances))}

    match = min(distance_dict, key=distance_dict.get)

    if keyword_dict[key].lower() in transcription_dict[match].lower() or transcription_dict[match].lower() in keyword_dict[key].lower() :
        correct += 1
    else :
        false += 1

    result_dict[keyword_dict[key]] = transcription_dict[match]

    key_min = min(distance_dict, key=distance_dict.get)
    min_val = distance_dict[key_min]
    key_max = max(distance_dict, key=distance_dict.get)
    max_val = distance_dict[key_max]


    thresholds = find_n_equidistant(n_thresholds, min_val, max_val)

    l_precision = []
    l_recall = []

    # then for each thresholds creating this way
    # divide the dict in two dict, one with items considered as positives
    # one with items considered as negatives

    word = transcription_dict[key]

    for threshold in thresholds:
        pos, neg = divide_dict(distance_dict, threshold)

        # use the dicts to get tp fp fn
        tp, fp = get_tp_fp(word, pos, transcription_dict)
        fn = get_fn(word, neg, transcription_dict)

        l_precision.append(precision(tp, fp))
        l_recall.append(recall(tp, fn))

    # average precision - area under the curve
    #auc = auc(l_recall, l_precision)
    #print(auc)
    label = word
    # plot the precision-recall curve
    plt.figure()
    #l_precision_new = [l_precision[i] for i in range(len(l_precision)) if l_recall[i] != None and l_precision[i] != None]
    #l_recall_new = [l_recall[i] for i in range(len(l_recall)) if l_recall[i] != None and l_precision[i] != None]

    #l_precision = [0 for i in l_precision if i == None]
    #l_recall = [0 for i in l_recall if i == None ]


    plt.plot(l_recall, l_precision, marker='.', label=label)
    # axis labels
    plt.xlabel('Average Recall')
    plt.ylabel('Average Precision')
    # show the legend
    plt.legend()
    # save the plot
    try :
        os.mkdir("result_plots/")
    except:
        pass

    savename = 'result_plots/'+label+'-result.png'
    plt.savefig(savename)
    plt.close()

    average_recall[:,i] = l_recall
    average_precision[:,i] = l_precision
    i+= 1


    # Display progress on dtw
    sys.stdout.write('\r')
    sys.stdout.write('{}/{} '.format(i,total))
    sys.stdout.flush()


overall_recall = np.mean(average_recall, axis=1)
overall_precision = np.mean(average_precision, axis=1)

# Print results
print('\n {}/{}'.format(correct,total))
print(result_dict)

# average precision - area under the curve
auc = auc(overall_recall, overall_precision)
print(auc)
# plot the precision-recall curve
plt.figure()
plt.plot(overall_recall, overall_precision, marker='.', label='our system')
# axis labels
plt.xlabel('Average Recall')
plt.ylabel('Average Precision')
# show the legend
plt.legend()
# save the plot
plt.savefig('result_plots/Average-result.png')
plt.close()


# Write result dict to file for inspection
 #with open('dtw_results_val.txt', 'w') as file:
    #for key in sorted(result_dict.keys()):
        #file.write(result_dict[key]+' '+key+'\n')
