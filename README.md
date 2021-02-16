# Packages for running the project

Here are the requirements to have on the running 
- python 3 with generic modules (numpy, matplotlib.pyplot, pandas, os, argparse, time)
- ffmpeg (command line tool, free and open source) for video edition in the accessible path.
- tensorflow and keras for neuron networks. 
- scikit-learn and scikit-image for image processing basic functions.
- pillow for image edition (draw on the image)
- click, for better command line tool

Project was coded and working under Linux Mint distribution. Also tested in macOS distribution (just make sure to have ffmpeg accessible in the PATH).

# Files explanation 

The project contains the following files: 
- **main.py**: it is the executable file.
- **object_extraction.py**: contains the functions to be called from the main to extract the different object
- **classifiers.py**: contains our 2 classifiers, to be called after object extraction
- **operator_classifier.py**: contains the functions to perform the classification among different operators
- **functions.py**: contains some utilities function, as shape restructing ones for instance. 
- **ccn.py**: file to train a CNN model. It loads the training dataset, aasemble it (if it is to merge different sources), and train the model. Also contains some code at the end to test the resulting classifier !
- **ccn_dataset_loading.py**: file to load the CHORME dataset, to extract only what we want, and to shape them in 28x28 (same as the MNIST). 

A few utilities
- **arial.ttf**: font used for image edition
- **cnn_classifier_with_operators**: the Keras model including '+' operator in the training datasets.
- **cnn_classifier_without_operators**: the Keras model only using MNIST data.


And our presentation: 
- **presentation.pdf**: slides of the presentation, explain the core functionnalities of the code

# What do we need to do

The input of the problem is **the video** of a robot that is moving above a map containing **numbers** and **mathematical operators**. The goal is quite simple: we must **construct and solve and equation from the path done by the robot**. 

The output that we must deliver is defined by the following point:
1. Must be a video, with the same frame rate and quality as the input, with writen on it 
    1. The current equation and it's solution
    2. The path done by the robot
2. The image detection must be robust against the following factors (for the digits and the operators)
    1. Translation 
    2. Rotation
    3. Colors

Here is what I think a list of task, of 'milestones', for this project. 
1. Detection and precise localisation of all the digits, using the first image
2. Detection and precise localisation of all the operators, using the first imag
2. Detection of the robot at all images (i.e. finding the path)
3. Class logic (datamodel) to construct and solve the equations
4. Rendering of the output video (already done with what I did...)


# Our solution, step by step

Here is an explanation of the solution that I implemented. I try to explain everything


## How to read the .avi file, and to create a new video

The solution is a work-around to use the skimage library using images and not video. Basically, we take each and every frame of the video and we create an image from it. This will help us to create an 'image collection' object using skimage functions. When the image analysis will be finished, we can easily reconstruct from this image collection a video. 

links that I have found useful to achieve this 
- https://scikit-image.org/docs/dev/user_guide/video.html
- https://trac.ffmpeg.org/wiki/ChangingFrameRate
- https://trac.ffmpeg.org/wiki/How%20to%20speed%20up%20/%20slow%20down%20a%20video

requirement: command line tool **ffmpeg** (https://www.ffmpeg.org/),  *free and open-source*.

command line to generate the images: 
`ffmpeg -i ../robot_parcours_1.avi -f image2 image%03d.png`

command lines to make the video from the images :
`ffmpeg -f image2 -framerate 2 -i image%03d.png video.avi` (create a video with good speed, but too high frame rate) 

The output video needs to have some text writen over the images, here is where I found how to do it (it make uses of pillow library): https://stackoverflow.com/questions/16373425/add-text-on-image-using-pil

## Object extraction 

Algorithm for object detection, using the first image 
1. Threshold to get only the forms on the picture
2. Remove the borders and the middle line using some filtering
3. Find the position of the robot using region.area filtering (region.area > 400)
4. Remove all the labels around this point to only keep regions with digits and operators.  
5. Merge labels that are spatially close ! 
6. Remove all the regions that have really small areas 

We end up with an label map containing only the digits and the operators, with as well the position of the robot ! It's now time to do some classification...

## Neuron Network for classification
- CNN Network with several **convolutions layers with small kermel** and **max pooling** with **batch normalization**, then **flattening**, then a dense **fully connected network** of 2 layers. 

## Datasets used for training and testing
I have joined two datasets for our application, in order to be really robust against rotation and against colors. It was quite long to join them, as I had problems of unbalanced classes, and of image compatibility (same size, same format...) 
- 60000 images of MNIST
- 10000 of CHORME containing only '+' and '='
followed by 
- Data augmentation with little rotations, translations, shearing and zooms 

## Classification rule
the classification rule isn't as straightforward as one might think. We make use of a few heuristic rules to make sure we have a really good classification. The algorithm goes as following 
1. Extract the 'features' from the first image. By features, I mean cropped images of every digit and of every operator. 
2. For each feature:
    1. Check if the feature is '%' by counting the number of group of pixels on this image. If it's 3, then it can only be this one. 
    2. If not, apply the feature as input to the trained CNN . Verify the heuristic rules (will be listed right after) and then attribute the label using the output of the network
    
## Some definitions for sake of understanding

- A **feature** is an image that has either a digit or an operator, with dimension 28x28 and with 3 pixels of paddings on each side
- The **classes** that we use are defined by an integer with the following order: 
`class_names = ['0','1','2','3','4','5','6','7','8','9','+','=','%']`
So for instance, if a feature is classifed as 11, it means it represents the operator '='