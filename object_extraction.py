# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:05:04 2020

The goal of this file is to create bounding boxes around all the shapes of the pictures. It contains all the 

@author: arthur
"""

import functions as fc 
import classifiers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import keras
import skimage, skimage.filters

from skimage import img_as_uint
from skimage.transform import rotate
from skimage.measure import label, regionprops
from skimage.color import rgb2gray

#%% Find the position of the robot 

def get_robot_position(image):
    """
    It returns the position of the robot from the rbg image given as input.

    Parameters
    ----------
    image :  RBG image as np.array
        Input image where to look for the objects.

    Returns
    -------
    Position of the robot.

    """
    image = rgb2gray(image)
    t = skimage.filters.threshold_otsu(image) + 0.05
    bw = image < t
    
    # label image regions
    label_image = label(bw)
    regions = regionprops(label_image)
    
    # remove from the labels the 'useless' ones
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox
        if maxr - minr > 200 or region.eccentricity > 0.99:
            coords = region.coords
            label_image[coords[:,0], coords[:,1]] = 0
    regions = np.array(regionprops(label_image))
    
    # Find the position of the robot
    robot_region = None
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox
        if region.area > 400:
            robot_region = region
            
    # Draw all of this on a single figure
# =============================================================================
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     ax.imshow(label_image, cmap='jet')
#     for region in regions:
#         minr, minc, maxr, maxc = region.bbox
#         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
#         ax.add_patch(rect)
# =============================================================================
    
    if robot_region == None: 
        return np.array([0,0])
    else: 
        center= np.array(robot_region.centroid)
        return center

#%% Find the position of each feature

def get_features_and_positions(image_rgb, rotation_angle = 0, classifier = 1, rotation_invariance = 1, class_names = []):
    """
    Returns the position of the objects in the given image, with their meaning (its class label)

    Parameters
    ----------
    image_rgb : RBG image as np.array
        Input image where to look for the objects.
    rotation_angle : float, optional
        Every image is randomly rotated by a angle within this range (for assessment of the robustness). The default is 0.
    classifier : int, optional, either 1 or 2
        Index of the classifier to use. The default is 1. 
    rotation_invariance : Bool, optional
        if 1 or 'True', then f-folds with 37 rotated images are averaged to classify more robustly. The default is 1.
    class_names : [String], optional
        Just for debugging. The default is [].

    Returns
    -------
    list
        position of the objects in the given image, with their meaning (its class label).

    """
    
    fc.plot_image(image_rgb)
    image = rgb2gray(image_rgb)
    t = skimage.filters.threshold_otsu(image) + 0.05
    bw = image < t
    fc.plot_image(bw, name = "debuging/bw.png")
    fc.plot_histogram(image)
    
    # label image regions
    label_image = label(bw)
    regions = regionprops(label_image)
    fc.plot_image(label_image, name = "debuging/label_image.png")
    
    # remove from the labels the 'useless' ones
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox
        if maxr - minr > 200 or region.eccentricity > 0.99:
            coords = region.coords
            label_image[coords[:,0], coords[:,1]] = 0
    regions = np.array(regionprops(label_image))
    
    # Find the position of the robot
    robot_region = None
    for i, region in enumerate(regions):    
        minr, minc, maxr, maxc = region.bbox
        if region.area > 400:
            robot_region = region
            
    # Remove from the label map the robot         
    region_centers = [np.array(r.centroid) for r in regions]
    robot_center = np.array(robot_region.centroid)
    distances_from_robot = np.array([np.linalg.norm(c-robot_center) for c in region_centers])
    regions_to_remove = regions[distances_from_robot < 60]
    for region in regions_to_remove:
        coords = region.coords
        label_image[coords[:,0], coords[:,1]] = 0
    regions = np.array(regionprops(label_image))
    
    # Merge labels that are really close 
    # (So we create a distance matrix, for all the left regions...) 
    region_centers = [np.array(r.centroid) for r in regions]
    distance_matrix = np.array([[np.linalg.norm(x-y) for y in region_centers] for x in region_centers])
    close_regions = np.logical_and(distance_matrix > 0 ,distance_matrix < 30)
    close_regions = np.triu(close_regions)
    regions_to_merge = np.argwhere(close_regions)
    for set in regions_to_merge:
        r1, r2 = regions[set[0]], regions[set[1]]
        coords = r1.coords
        label_image[coords[:,0], coords[:,1]] = r2.label
    regions = np.array(regionprops(label_image))
    
    # Remove the regions with not enough points
    for i, region in enumerate(regions):    
        if region.area < 40:
            coords = region.coords
            label_image[coords[:,0], coords[:,1]] = 0
    regions = np.array(regionprops(label_image))
    
    # Draw all of this on a single figure
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(label_image, cmap='jet')
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    # Extract the images (and check if it is blue or not)
    fig, axs = plt.subplots(1, len(regions))
    imgs = []
    imgs_rgb = []
    blue_images = np.zeros(len(regions))
    for i, region in enumerate(regions):
        img = fc.rescale_feature(region.image)
        bb = region.bbox
        img_rgb = image_rgb[bb[0]:bb[2],bb[1]:bb[3],:]
        if fc.blue_detect( image_rgb[region.coords[:,0], region.coords[:,1], :] ):
            blue_images[i] = 1
        imgs.append(img)
        imgs_rgb.append(img_rgb)
        axs[i].imshow(img)
        axs[i].axis('off')
    imgs = np.array(imgs)
    np.save("imgs", imgs)
    
    # Randomly rotate the images obtained (for rotation invariance)
    for i, img in enumerate(imgs):
        angle = (np.random.random()-0.5)*rotation_angle
        imgs[i] = rotate(img,angle)        
    
    # Classification 
    print("- classification with classifier {}".format(classifier))
    if classifier == 1:
        model = keras.models.load_model('cnn_classifier_with_operators')    
        classes = classifiers.classifier_1(imgs, model = model, rotation_invariance = rotation_invariance)
    if classifier == 2:
        model = keras.models.load_model('cnn_classifier_without_operator')    
        classes = classifiers.classifier_2(imgs, blue_images, model = model, rotation_invariance = rotation_invariance)
        
    # Plotting classification results 
    fig, axs = plt.subplots(1, len(imgs))
    for i, img in enumerate(imgs):
        axs[i].imshow(img[:,:])
        axs[i].axis('off')
        title = '{}'.format(class_names[int(classes[i])])
        axs[i].set_title(title)
    fig.savefig("debuging/objects.png")
    
    return [(np.array(r.centroid),int(c)) for r,c in zip(regions, classes)]
    


