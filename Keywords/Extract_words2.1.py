# Import packages
from svg.path import parse_path, Line
from skimage.io import imread
from xml.dom import minidom
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
from skimage import (feature, color, measure, util)
from skimage.transform import resize
from scipy.signal import convolve2d


def extract_polygons(svg_path):
    '''
    Extract polygon coordinates from an svg file that describes their path
    '''

    doc = minidom.parse(svg_path)
    path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    path_id = [path.getAttribute('id') for path in doc.getElementsByTagName('path')]
    doc.unlink()

    # Make dictionary of lines for each box
    word_positions = {}
    for element in range(len(path_id)):
        id = path_id[element]
        path = path_strings[element]

        # Get array of points and save to dictionary
        instructions = path.split(" ")
        letters = list(range(0, len(instructions), 3))
        for index in sorted(letters, reverse=True):
            del instructions[index]
        instructions = list(map(float, instructions))
        instructions = np.array(instructions).reshape(-1, 2)
        word_positions[id] = instructions

        print("polygons", id)

    return word_positions

def apply_mask(img, word_positions):

    '''
    Applies a mask to an image so as to remove everything else than what is contained inside a polygon or list of polygons

    Inputs:
    img = full image array
    points = list of tuples of coordinates of the polygons

    Outputs:
    list of image arrays, one corresponding to polygon
    '''

    words_per_image = {}

    for key, value in word_positions.items():

        # Get the points of the polygon from the dictionnary
        points = value
        points = points.astype(int)
        points = tuple([tuple(row) for row in points])

        # Draw the polygon on a mask
        mask = Image.new('L', (img.shape[1], img.shape[0]))
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        mask = np.array(mask)

        # Apply the mask to the image
        newImageArray = np.empty(img.shape, dtype=np.uint8)
        newImageArray = img
        newImageArray = img*mask
        newImageArray = newImageArray.astype(np.uint8)

        # Save into a dictionary
        words_per_image[key] = newImageArray
        print("masked", key )
    return words_per_image

def save_words_per_image(words_per_image, save_dir):
    '''
    Saves images with their correct ID
    '''
    for id in words_per_image.keys():
        image = (words_per_image[id]*255).astype(np.uint8)
        image = Image.fromarray(image)
        save_name = save_dir+id+'.png'
        image.save(save_name)

def compute_features(image):
    features = np.zeros((image.shape[1],4))
    pixel_nr = np.count_nonzero(image, axis=0)/image.shape[1]
    features[:,0] = pixel_nr
    filt = np.array([-1,1])
    filt = np.reshape(filt,(2,1))
    nr_of_transitions = np.count_nonzero((convolve2d(image,filt,mode="valid"))**2,axis=0)
    features[:,1]=nr_of_transitions
    crap = np.flip(image, axis=1)
    features[:,2] = np.argmax(image,axis=0)
    features[:,3]=image.shape[1]-np.argmax(crap,axis=0)
    colmeans = np.mean(features,axis=0)
    colsd = np.std(features,axis=0)
    return (features-colmeans)/colsd


if __name__=='__main__':

    # Set directories
    rootdir = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/PatRec17_KWS_Data-master'
    svgdir = rootdir+'/ground-truth/locations/'
    jpgdir = rootdir+'/images/'
    savedir = rootdir+'/cut_images_test/'

    try:
        os.mkdir(savedir)
    except:
        pass

    # Do the polygons and binarization need to be applied ?
    first_step = False

    ### POLYGON APPLICATION AND BINARIZATION
    if first_step :
        img_paths = []
        save_names = []

        # Import all images files
        for filename in sorted(os.listdir(jpgdir)):
            if filename.endswith(".jpg"):
                img_paths.append(jpgdir+filename)
                save_name = filename.split(".")
                save_names.append(save_name[0])

        # Import all svg files
        svg_paths = []
        for filename in sorted(os.listdir(svgdir)):
            if filename.endswith(".svg"):
                svg_paths.append(svgdir+filename)

        total_images = len(save_names)

        for i in range(len(save_names)):
            svg_path = svg_paths[i]
            total_words = len(svg_path[i])
            img_path = img_paths[i]
            save_path = savedir+str(save_names[i])+'/'

            try:
                os.mkdir(save_path)
            except :
                pass

            # Import image and corresponding svg file and set borders to white
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image[0:50,:] = 255
            image[image.shape[0]-50:,:] = 255
            image[:,0:50] = 255
            image[:, image.shape[1]-50:] = 255

            # Get polygons and corresponding id's
            word_positions = extract_polygons(svg_path)

            # Binarize image
            image = filters.threshold_otsu(image)>image
            image = image.astype(int)
            #image = (image*255).astype(np.uint8)
            #img = Image.fromarray(image)

            # Use polygon coordinates to extract each word from image
            words_per_image = apply_mask(image, word_positions)
            # Save the images of  each word
            save_words_per_image(words_per_image, save_path)
            print("saved", i)
    ########################################


    dir_list = ['270', '271','272','273','274','275','276','277','278','279','300','301','302','303','304']
    savedir = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/PatRec17_KWS_Data-master/cut_images_test/'

    for pic in dir_list:
        path = savedir+pic+'/'
        for file in os.listdir(path):

            img_path = path+file
            image = imread(img_path)
            trial = np.nonzero(image)

            # Crop image and resize
            cropped = image[min(trial[0]):max(trial[0])+1,min(trial[1]):max(trial[1])+1]
            cropped = resize(cropped,(100,100), anti_aliasing=True)

            ### FEATURE EXTRACTION #####################
            save_dir = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/featurestest/'

            try:
                os.mkdir(save_dir)
            except:
                pass

            save_dir = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/featurestest/'+pic+'/'

            try:
                os.mkdir(save_dir)
            except:
                pass

            save_path = save_dir+file

            # Compute features
            feature_vector = compute_features(cropped)
            if np.sum(np.isnan(feature_vector))>0:
                feature_vector = np.nan_to_num(feature_vector)
            np.savetxt(save_path+'.txt', feature_vector, delimiter=",")
            print("features", file)
            ############################################################
    print("done")
