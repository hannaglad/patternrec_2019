from svg.path import parse_path
from svg.path.path import Line
from xml.dom import minidom
import cv2
import opencv_wrapper as cvw
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from PIL import Image, ImageDraw
import os
import sys

rootdir = '/home/hanna/Documents/UNIFR/2_semester/Pattern_recognition/Exercises/keywords/PatRec17_KWS_Data-master'
svgdir = rootdir+'/ground-truth/locations/'
jpgdir = rootdir+'/images/'
savedir = rootdir+'/cut_images/'

# Get all the files etc etc

img_paths = []
save_names = []
for filename in sorted(os.listdir(jpgdir)):
    if filename.endswith(".jpg"):
        img_paths.append(jpgdir+filename)
        save_name = filename.split(".")
        save_names.append(save_name[0])

svg_paths = []
for filename in sorted(os.listdir(svgdir)):
    if filename.endswith(".svg"):
        svg_paths.append(svgdir+filename)



for i in range(len(save_names)):
    svg_path = svg_paths[i]
    img_path = img_paths[i]
    save_path = savedir+str(save_names[i])+'/'

    try:
         os.mkdir(save_path)
    except :
        pass

    # Import image
    image = cv2.imread(img_path)
    img = Image.open(img_path).convert("RGB")

    # Import svg
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



    #progress report
    total_words = len(word_positions.keys())
    i = 0
    # Create polygon masks for each image
    for key, value in word_positions.items():
        i+=1
        #mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        points = value
        name = save_path+str(key)+'.png'
        points = points.astype(int)

        points = tuple([tuple(row) for row in points])

        mask = Image.new('1', (image.shape[1], image.shape[0]),0)
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        mask = np.array(mask)

        newImageArray = np.empty(image.shape, dtype=np.uint8)


        newImageArray[:,:,:3] = image[:,:,:3]

        newImageArray[:,:,0] = newImageArray[:,:,0]*mask
        newImageArray[:,:,1] = newImageArray[:,:,1]*mask
        newImageArray[:,:,2]= newImageArray[:,:,2]*mask

        result = Image.fromarray((newImageArray).astype(np.uint8))
        result.save(name)

        sys.stdout.write('\r')
        sys.stdout.write('{}/{} words extracted'.format(i,total_words))
        sys.stdout.flush()
    sys.stdout.flush()
