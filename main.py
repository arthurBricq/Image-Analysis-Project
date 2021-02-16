#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 09:57:54 2020

Project's entry point (command line tool of the project within this file). 
It is the file that will be called when running the project

@author: arthur
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr
import time 
import click 


import numpy as np
import skimage, skimage.io
import object_extraction 
import functions as fc 

from PIL import Image, ImageDraw, ImageFont

#%% Argument parser to read the video file properly

import argparse
parser = argparse.ArgumentParser(description='Command line tool for the Image Analysis\'s special project')
parser.add_argument('--input', help='Path to the input video', required=False,
                    default = 'robot_parcours_1.avi')
parser.add_argument('--output', help='Path to the output video (will display the results)',
                    required=False, default = 'test_output.avi')
parser.add_argument('--classifier', help='Which classifier to use. 1 stands for the robust one. 2 stands for the unrobust one. ', 
                    required=False, default = 1)
parser.add_argument('--ffolds', help='0 if no rotation invariance to be applied. 1 if to be applied (default)',
                    required=False, default = 1)

args = parser.parse_args()
input = args.input
output = args.output
classifier = int(args.classifier)
rotation_invariance = int(args.ffolds)

start_time = time.time()

print("")
print("--------------------------------")
print("* Image analysis final project *")
print("--------------------------------")
print("\n Authors: ")
print("* Arthur Bricq")
print("* Anne-Aimée Bernard")
print("* Jonas Ulbrich")

print("\n Recap of the parameter used.")
print("* classifier used: {}    ".format(classifier) + ("(color invariant)" if classifier == 1 else "(make use of the color information)"))
print("* rotation invariance more robust using f-folds ?    " + ("YES" if rotation_invariance else "NO"))

# %% Remove previous file to make sure it's all ok

os.system("rm {}".format(output))
os.system('rm -r images')
os.system('rm -r debuging')
os.system('rm -r images_annoted')

# %% Generation of Image Sequence for easier work after 

os.system('mkdir images')
os.system('mkdir debuging')
os.system('mkdir images_annoted')
os.system('ffmpeg -hide_banner -loglevel panic -i {} -f image2 images/image%03d.png'.format(input))

#%% Creation of Image Collection 

# Load the images
number_of_images = len([name for name in os.listdir('images')])
im_names = ['image{:03d}.png'.format(i) for i in np.arange(1,number_of_images)]
file_names = ['images/' + name for name in im_names]
ic = skimage.io.imread_collection(file_names)
images = skimage.io.concatenate_images(ic)
print("\n Loading images from the intput video...")
print('- Number of frames: ', images.shape[0])
print('- Image size: {}, {} '.format(images.shape[1], images.shape[2]))
print('- Number of color channels: ', images.shape[-1])

#%% Analysis of the first image 

print("\n Analysing the first image, to extract position of the digits and operators.")
class_names = ['0','1','2','3','4','5','6','7','8','9','+','=','%','-','*']
image = images[0]
# image = skimage.io.imread("arena-shapes-03.png")
features = object_extraction.get_features_and_positions(image, 
                                                        rotation_invariance = rotation_invariance, 
                                                        class_names = class_names, 
                                                        classifier = classifier)
                
#%% Analysis of the robot's path
print("\n Analysing the robot's path through the frames.")

font = ImageFont.truetype("arial.ttf", 40)
last_visited_index = -1
string_to_display = "Equation:   "
visited_features = []
visited_positions = []
last_robot_position = None
for image, name in zip(images, im_names): 
    # for every image, will compute the distance between the robot and all the features on the image
    robot_position = object_extraction.get_robot_position(image)
    distances_to_features = [np.linalg.norm(robot_position-feature[0]) for feature in features ]
    if min(distances_to_features) < 15: 
        # if distance is small, the robot is above a feature. 
        index = np.argmin(distances_to_features)
        if index != last_visited_index:
            # just make sure it's not the same as the last one. 
            last_visited_index = index 
            feature = class_names[features[index][1]]
            visited_features.append(feature)
            string_to_display += feature
            if feature == '=':
                tmp = visited_features
                string_to_display = fc.solve_the_equation(visited_features, string_to_display)
                print("Equation has been solved ! ")
            
    # Plot on the images the path and the equation 
    image = Image.open('images/'+name)
    ImageDraw.Draw(image).text((135, 350),string_to_display,(255,255,255),font=font)
    if last_robot_position is not None: 
        # compute the distance between the last and the new one
        distance = np.sqrt((last_robot_position[0]-robot_position[0])**2 + (last_robot_position[1]-robot_position[1])**2 )
        # sometimes the distance fuck up 
        if distance < 50 and distance > 0:
            visited_positions.append(tuple(robot_position[::-1]))
            
    if len(visited_positions) >= 2:
        for i in range(len(visited_positions)-1):
            ImageDraw.Draw(image).line([visited_positions[i], visited_positions[i+1]], fill ="red", width = 2)

    # Save the computed data for next round 
    image.save('images_annoted/' + name)
    last_robot_position = robot_position   
    
    
# %% Create the video from an a folder of pictures

print("\n Start saving the video output file.")
os.system("ffmpeg -hide_banner -loglevel panic -f image2 -framerate 2 -i images_annoted/image%03d.png {}".format(output))

print("\n End of execution !")
print("- time of execution: {:.02f} [s]".format(time.time() - start_time))

if click.confirm("\n Open the video ? "):
    os.system("xdg-open {}".format(output))
else:
    print("Thank you for your time !")

