# DTW

from fastdtw import fastdtw
from scipy.spatial import distance
import numpy as np
import os
import sys
from multiprocessing import Process, Pool
from functools import partial

euclidean = distance.euclidean

############### IMPORT DATA
# Load all of the feature vectors
root = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/featurestest/'
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
# define valid keys
validKeys = []
for key in keyword_dict.keys() :
    if any(str(train_image) in key for train_image in train):
        validKeys.append(key)

#transcription_dict_ref = {key:value for key,value in transcription_dict.items() if key in validKeys}
total = len(keyword_dict.keys())

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

    # Find the smallest distance
    match = min(distance_dict, key=distance_dict.get)

    if keyword_dict[key].lower() in transcription_dict[match].lower() or transcription_dict[match].lower() in keyword_dict[key].lower() :
        correct += 1
    else :
        false += 1

    result_dict[keyword_dict[key]] = transcription_dict[match]

    i+=1

    # Display progress on dtw
    sys.stdout.write('\r')
    sys.stdout.write('{}/{} '.format(i,total))
    sys.stdout.flush()

# Print results
print('\n {}/{}'.format(correct,total))
print(result_dict)

# Write result dict to file for inspection
with open('dtw_results_val.txt', 'w') as file:
    for key in sorted(result_dict.keys()):
        file.write(result_dict[key]+' '+key+'\n')
