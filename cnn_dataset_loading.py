#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:06:17 2020
@author: arthur
"""
import pandas as pd 
import numpy as np
import functions as fc 
import tensorflow as tf

#%% Read the data and display the dataset 

test = pd.read_pickle("outputs/test/test.pickle")
train = pd.read_pickle("outputs/train/train.pickle")
df_test = pd.DataFrame(test)
df_train = pd.DataFrame(train)

df_train['label'] = df_train['label'].apply(lambda x: np.where(x)[0][0])
df_test['label'] = df_test['label'].apply(lambda x: np.where(x)[0][0])
print("Unique elements in the dataset: ", np.unique(df_train.label))

labels = ['(',')','+','-','0','1','2','3','4','5','6','7','8','9','=','[',']']
for i, l in enumerate(labels):
    count = np.count_nonzero(df_train['label']==i)
    print(" - {}: has {} elements".format(l,count))
    
#%% Construct the training dataset
    
labels_of_interest = [2]
n_train = 800
n_test = 70
train_images, train_labels = [], []
test_images, test_labels = [], []
for l in labels_of_interest:
    
    tmp_train = df_train[df_train['label'] == l]
    
    imgs_train = [img for img in tmp_train.features]
    lbls_train = [10 if l == 2 else 11 for _ in tmp_train.label]
    print(np.array(imgs_train).shape)
    train_images.append(imgs_train)
    train_labels.append(lbls_train)
    
    tmp_test = df_test[df_test['label'] == l]
    imgs_test = [img for img in tmp_test.features]
    lbls_test = [10 if l == 2 else 11 for _ in tmp_test.label]
    test_images.append(imgs_test)
    test_labels.append(lbls_test)

train_images = np.concatenate(train_images, axis = 0)
train_labels = np.concatenate(train_labels, axis = 0)
test_images = np.concatenate(test_images, axis = 0)
test_labels = np.concatenate(test_labels, axis = 0)


#%% pre-processing function 

def process_image(img):
    padding = 4 
    c = img.shape[0]
    image = np.zeros((c+2*padding,c+2*padding))
    image[padding:-padding, padding:-padding] = 1-img
    image = tf.constant(image)
    image = image[tf.newaxis, ..., tf.newaxis]
    image = tf.image.resize(image, [28,28])[0,...,0].numpy()
    return image

#%% Before being saved, the images need to be preprocessed a little to fit the "convention" of MNIST

train_images_ = []
test_images_ = []
for i, img in enumerate(train_images):
    train_images_.append(process_image(img))
for i, img in enumerate(test_images):
    test_images_.append(process_image(img))
train_images_ = np.array(train_images_)
test_images_ = np.array(test_images_)

#%% Let's add some padding to the images

fc.plot_image(img) 
padding = 4 
c = img.shape[0]
image = np.zeros((c+2*padding,c+2*padding))
image[padding:-padding, padding:-padding] = 1-img
fc.plot_image(image) 
    

#%% And resize it to the good size !
        
fc.plot_image(img) 
image = tf.constant(img)
image = image[tf.newaxis, ..., tf.newaxis]
image = tf.image.resize(image, [28,28])[0,...,0].numpy()
fc.plot_image(image) 


#%% Then save the images

np.save("x_train_ops.npy",train_images_)
np.save("x_test_ops.npy",test_images_)
np.save("y_train_ops.npy",train_labels)
np.save("y_test_ops.npy",test_labels)











