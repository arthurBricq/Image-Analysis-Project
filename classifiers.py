#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:12:14 2020

@author: arthur
"""

import numpy as np 
import functions as fc 
import operator_classifier

from skimage.transform import rotate

def classifier_1(features, model, rotation_invariance = True):
    """
    Given an array of features (objects of the image with the proper dimension), returns an array of classes describing each one of them.
    
    This method uses the robust CNN to make the classification. 
    - rotation invariant
    - color invariant  (NOT USING HE COLORS)
    - translation invariant 
    

    Parameters
    ----------
    features : Array of images 
        Each image describes an object to be classified
    model : Tensorflow.Keras.model 
        CNN Model to use for the classification (had to be trained before)
    rotation_invariance : Bool, optional
        True if it is to apply f-folds on the image when classifying. The default is True.


    Returns
    -------
    Array of classes
    """
    
    classes = np.zeros(features.shape[0])

    # 1. Check if the image is the '%' operator
    for i, img in enumerate(features): 
        number_of_region = fc.get_number_of_regions(img)
        if number_of_region == 4:
            classes[i] = 12
        if number_of_region == 3: 
            classes[i] = 11 

    # 2. For the rest of the features, apply the CNN network  
    
    # a. Apply a serie of rotation to each feature 
    angles = np.arange(0,370 if rotation_invariance else 10,10)
    imgs = []
    for i, img in enumerate(features):
        rotated_images = np.array([rotate(img,angle)>0.1 for angle in angles])
        imgs.append(fc.normalize_images(rotated_images))

    # b. Make average of predictions to find the real feature
    for i, rotated_images in enumerate(imgs):
        if classes[i] == 12: 
            continue
        if classes[i] == 11: 
            continue
        all_predictions = [] 
        for img in rotated_images:
            test_image = np.expand_dims(img, axis=0)
            predictions = model.predict(test_image)
            c = np.argmax(predictions)
            p = np.max(predictions)
            all_predictions.append(predictions)
        mean_prediction = np.mean(all_predictions, axis = 0)
        p = np.max(mean_prediction)
        c = np.argmax(mean_prediction)
        if (c == 10 and p < 0.5) or (c == 2 and p < 0.33):
            # it means it's a seveclasify_features(features, model)n
            c = 7         
        if (c == 8 and p < 0.35):
            c = 11
        if (c==10):
            # either + or *, let's find out using our second classifier
            c, _ = operator_classifier.classify_bin_operator(features[i])
            
        classes[i] = c 
        print("- image {} classified as {}  with probability  {:.02f}".format(i,c,p))
        
    return classes


def classifier_2(features, blue_images, model, rotation_invariance = True):
    """
     Given an array of features (objects of the image with the proper dimension), returns an array of classes describing each one of them.
    
    This method uses the CNN to make the classification, and also uses the colors to classify.
    - rotation invariant
    - translation invariant 
    

    Parameters
    ----------
    features : Array of input images to be classified
    blue_images : Array of booleans
        Is true at any position if the same image is in blue or not
    model : CNN model to use
    rotation_invariance : Bool, optional
        True if it is to apply f-folds on the image when classifying. The default is True.

    Returns
    -------
    classes : 
    Array of classes

    """
    classes = np.zeros(features.shape[0])
    
    # 1. Classify the operator
    for i, img in enumerate(features): 
        if not blue_images[i]: 
            continue
        c, _ = operator_classifier.classify_bin_operator(img)
        classes[i] = c  
            
    # 2. Classify the number using our trained neuron network for this case   
    angles = np.arange(0,370 if rotation_invariance else 10,10)
    imgs = []
    for i, img in enumerate(features):
        rotated_images = np.array([rotate(img,angle)>0.1 for angle in angles])
        imgs.append(fc.normalize_images(rotated_images))
            
    for i, rotated_images in enumerate(imgs):
        if blue_images[i]:
            continue
        all_predictions = [] 
        for img in rotated_images:
            test_image = np.expand_dims(img, axis=0)
            predictions = model.predict(test_image)
            c = np.argmax(predictions)
            p = np.max(predictions)
            all_predictions.append(predictions)
        # classification rule is here
        mean_prediction = np.mean(all_predictions, axis = 0)
        p = np.max(mean_prediction)
        c = np.argmax(mean_prediction)
        # heuristic rules are here
        if (c == 2 and p < 0.33) or (c == 8 and p < 0.41):
            # it means it's a seven
            c = 7         
        classes[i] = c 
        print("- image {} classified as {}  with probability  {:.02f}".format(i,c,p))
        
    return classes
        

    

