# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:07:00 2020

This file host a few functions to be used in development.


@author: arthur
"""

import os
import numpy as np 
import skimage, skimage.io
import matplotlib.pyplot as plt 

from skimage import img_as_uint
from skimage.measure import label
from skimage import img_as_bool, img_as_int
from skimage.transform import resize

#%% Utilities 

def load_images():
    number_of_images = len([name for name in os.listdir('images')])
    im_names = ['image{:03d}.png'.format(i) for i in np.arange(1,number_of_images)]
    file_names = ['images/' + name for name in im_names]
    ic = skimage.io.imread_collection(file_names)
    images = skimage.io.concatenate_images(ic)
    print('Number of images: ', images.shape[0])
    print('Image size: {}, {} '.format(images.shape[1], images.shape[2]))
    print('Number of color channels: ', images.shape[-1])
    return images

def plot_image(image, name = ""):
    """
    Plot the image.
    If a name is given, it will save the image at the given name.
    """
    plt.figure()
    plt.imshow(image, cmap='jet')
    if name != "" : 
        # then save the image at the given name
        plt.savefig(name)
        # skimage.io.imsave(name, img_as_uint(image))

        
    
def plot_histogram(image, name = ""):
    """
    Plot the histogram of the image
    """
    plt.figure()
    plt.hist(image.ravel(), bins=256, histtype='step', color='black')
    
    
# %% Color detector
    
def blue_detect(object_px,primary_color_tol=0.24,secondary_color_tol=10,rgb=True,thresholding=False,blue_tone=[],thld_tol=10):
    ''' takes an object and checks if blue
        @param object_px            array containing the RGB values for the n
                                    pixels of the object [n x 3]
        @param thresholding         boolean to change mode of identification. If
                                    false the color blue will be identified
                                    based on the values of each channel
                                    --> blue means r<g<b, if r<g~b cyan
                                    if thresholding=True then the object is blue
                                    if the average lays in the intrvall
                                    blue_tone +/- thld_tol
        @param p/s_color_tol        The evaluation formula is the following:
                                    abs(r-g)<secondary_color_tol
                                    and secondary_color*(1+primary_color_tol)<=b
                                    NOTE that primary_color_tol is in percentage
                                    and secondary_color_tol is a grey scale
                                    value in [0:255].
        @param rgb                  False for pixels stored in BGR
        @return                     True if blue, else False
    '''

    nx,ny=object_px.shape
    #this line may give bad results for objects of size n=3 if not used correctly
    avg_color=np.mean(object_px,axis=0) if ny==3 else np.mean(object_px,axis=1)

    if rgb:
        r=avg_color[0]
        g=avg_color[1]
        b=avg_color[2]
    else: #BGR
        r=avg_color[2]
        g=avg_color[1]
        b=avg_color[0]

    if thresholding:
        if rgb:
            r_ref=blue_tone[0]
            g_ref=blue_tone[1]
            b_ref=blue_tone[2]
        else: #BGR
            r_ref=blue_tone[2]
            g_ref=blue_tone[1]
            b_ref=blue_tone[0]

        if r_ref-thld_tol<=r<=r_ref+thld_tol and g_ref-thld_tol<=g<=g_ref+thld_tol and b_ref-thld_tol<=b<=b_ref+thld_tol:
            return True
        else:
            return False


    else:
        #secondary color=r ->blue/violet else blue/cyan
        secondary_color=r if r>g else g

        if np.abs(r-g)<secondary_color_tol and secondary_color*(1+primary_color_tol)<=b:
            return True
        else:
            return False



#%% Functions for the Neuron Network --> classification

def rescale_feature(img):
    """
    Given an input image, return an output image with the same dimensions as the MNIST dataset. 
    If the input image is not square, it will put it in a sClipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    If the input image is square, it will simply reshape it to match the size

    Parameters
    ----------
    img : [[int]]
        Input image to be rescaled.

    Returns
    -------
    rescaled_img : TYPE
        Output image, can be passed as input of the neuron network.    


    """
    h, w = img.shape         
    if w != h: 
        # Fit in square with some padding
        side = max(h, w)
        padding = 3 
        new_img = np.zeros((side+2*padding,side+2*padding))
        if h > w: 
            new_img[padding:-padding,padding+int(side/2)-int(w/2):padding+int(side/2)-int(w/2)+w] = img
        if w > h: 
            new_img[padding+int(side/2-int(h/2)):padding+int(side/2-int(h/2))+h,padding:-padding] = img
    else:
        new_img = img
    
    # Rescale so that the image ends up with the right size: 28*28 
    rescaled_img = img_as_bool(resize(new_img, (28, 28)))
    return rescaled_img  


def rescale_feature_rgb(img):
    """
    This function apply the rescale_feature function to an RGB image
    """
    new_img = np.zeros(shape = (28, 28, 3))
    new_img[:,:,0] = rescale_feature(img[:,:,0])
    new_img[:,:,1] = rescale_feature(img[:,:,1])
    new_img[:,:,2] = rescale_feature(img[:,:,2])
    return new_img
    
    

def normalize_images(X):
    """
    Set up the image to become input of the neuron network

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    X_n : TYPE
        DESCRIPTION.

    """
    X_n = X.reshape(X.shape[0],28,28,1)
    X_n = X_n.astype('float32')
    return X_n

def get_number_of_regions(img):
    """
    Returns the number of group of pixels in the image
    """
    # figure is % <==> contains 4 labels on it 
    label_image = label(img)
    count = np.unique(label_image).size
    return count
    

#%% function to solve the equation  

# (python changes the array used in the function)

def solve_the_equation(features, eq_as_string = "0"):
    """
    Call this function to solve the equation.

    Parameters
    ----------
    features : array of features
        It's the array of all the extracted features (digits or operatos).
    eq_as_string : String, optional
        It's the equation to solve as a string. The default is "0".

    Returns
    -------
    String
        The solved equation, as a string.

    """
    
    def apply_operator(function, index):
        """
        Helper function to solve the equation. 
        It applies the operator to the first and the third element of the array features
        """
        # print(index)
        # print(features)
        try:
            n1 = float(features[index-1])
            n2 = float(features[index+1])
            results = function(n1,n2)
            features[index-1] = results
            del features[index]
            del features[index]
            return True
        except ValueError:
            return False 
        
    def operate(index):
        # print(index)
        operation = features[index]
        # find all the operators 
        if operation == '+':
                has_worked = apply_operator(lambda x,y: x + y, index)
                if not has_worked:
                    print("Error in the format. The equation can't be solved. ")
                    return False
        if operation == '%':
                has_worked = apply_operator(lambda x,y: x / y, index)
                if not has_worked:
                    print("Error in the format. The equation can't be solved. ")
                    return False
        if operation == '*':
                has_worked = apply_operator(lambda x,y: x * y, index)
                if not has_worked:
                    print("Error in the format. The equation can't be solved. ")
                    return False
        if operation == '=':
                del features[1]
                return True
            
    
    print("Solving the equation : " + eq_as_string)

    # at this point, we assume that the array features alternate between 
    # digis and operators, and that the last operator is '='
   
    # while len(features)-1 and counter < 100:
    print(features)
    
    # priority given to '%' and '*', from left to right 
    while np.count_nonzero([1 if t == '%' or t == '*' else 0 for t in features]):
        priority = np.where([1 if t == '%' or t == '*' else 0 for t in features])[0][0]
        operate(priority)
        
    # then, it goes for '+' and '-'
    while np.count_nonzero([1 if t == '+' or t == '-' or t == '=' else 0 for t in features]):
        priority = np.where([1 if t == '+' or t == '-' or t == '=' else 0 for t in features])[0][0]
        operate(priority)
    
    return eq_as_string + str(features[0])