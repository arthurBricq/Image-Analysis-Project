#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:03:57 2020

In this file, I try a cnn network for robust classification

Inspired by https://medium.com/datadriveninvestor/image-processing-for-mnist-using-keras-f9a1021f6ef0

@author: arthur
"""

import keras
import numpy as np 
import matplotlib.pyplot as plt
import functions as fc 
import classifiers 

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Conv2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import rotate
from sklearn.utils import class_weight

#%% Dataset construction 

(X_train, y_train) , (X_test, y_test) = mnist.load_data()

# =============================================================================
# X_train_2 = np.load("x_train_ops.npy") * 255
# y_train_2 = np.load("y_train_ops.npy")
# X_test_2 = np.load("x_test_ops.npy") * 255
# y_test_2 = np.load("y_test_ops.npy")
# 
# X_train = [X_train_1, X_train_2]
# X_train = np.concatenate(np.array(X_train))
# X_test = [X_test_1, X_test_2]
# X_test = np.concatenate(np.array(X_test))
# 
# y_train = [y_train_1, y_train_2]
# y_train = np.concatenate(np.array(y_train))
# y_test = [y_test_1, y_test_2]
# y_test = np.concatenate(np.array(y_test))
# =============================================================================

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%% 

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# Normalizing the inputs of the network
X_train= X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_train/=255

X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_test = X_test.astype('float32')
X_test/=255

y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

#%% CNN Network

classifier = Sequential()

# cnn 1 (32 filters)
classifier.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
BatchNormalization(axis=-1) #Axis -1 is always the features axis
classifier.add(Activation('relu'))

# cnn 2 (32 filters)
classifier.add(Conv2D(32, (3,3)))
BatchNormalization(axis=-1)
classifier.add(Activation('relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
BatchNormalization(axis=-1)

# cnn 3 (64 filters)
classifier.add(Conv2D(64, (3,3)))
BatchNormalization(axis=-1)
classifier.add(Activation('relu'))

# cnn 4 (64 filters)
classifier.add(Conv2D(64, (3,3)))
classifier.add(Activation('relu'))

# max pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

# flattening
classifier.add(Flatten())
BatchNormalization()

# fully connected neural network 
classifier.add(Dense(512))
BatchNormalization()
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(10))
classifier.add(Activation('softmax'))

classifier.summary()
#%% 

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_gen = ImageDataGenerator(rotation_range=20, 
                                width_shift_range=0.08, 
                                shear_range=0.3, 
                                height_shift_range=0.08, 
                                zoom_range=0.01)

training_set = train_gen.flow(X_train, y_train, batch_size=64)
test_set = train_gen.flow(X_test, y_test, batch_size=64)
classifier.fit_generator(training_set, 
                         class_weight = class_weights,
                         steps_per_epoch=X_train.shape[0]//64, 
                         validation_data= test_set, 
                         validation_steps=X_test.shape[0]//64, 
                         epochs=5)

# Saving it to something

classifier.save('cnn_classifier_without_operator')

#%% Robustness method : some sort of cross validation with different rotated images

# Heuristic rules used for robust classification
# - classified as '10' with low prob --> it's a seven
# - classified as '2' with really low prob --> it's a seven 
# - classified as '10' with high prob --> it's either a '+' or a '*'
    
class_names = ['0','1','2','3','4','5','6','7','8','9','+ or *','=','%']
features = np.load("imgs.npy")

# load the CNN to use
model = keras.models.load_model('cnn_classifier_without_operator')

# Randomly rotate the images obtained (rotation invariance)
for i, img in enumerate(features):
    angle = (np.random.random()-0.5)*0
    features[i] = rotate(img,angle)

# classes = classifiers.classifier_1(features, model = model, rotation_invariance = True)
classes = classifiers.classifier_2(features, blue_images = np.zeros(len(features)) , model = model, rotation_invariance = False)


# Print the results 
fig, axs = plt.subplots(1, len(features))
for i, img in enumerate(features):
    axs[i].imshow(img[:,:])
    axs[i].axis('off')
    title = '{}'.format(class_names[int(classes[i])])
    axs[i].set_title(title)
    
#%% Testing out a little thing about rotation (and it's fine)
    
features = np.load("imgs.npy")#[2:6,:,:]

# 1. For every image, construct a serie of this image rotated at several angles
# Also think about normalising them.
angles = np.arange(-100,110,50)
fig, axs = plt.subplots(features.shape[0],angles.shape[0])
for i, img in enumerate(features):
    rotated_images = np.array([rotate(img,angle)>0.1 for angle in angles])
    for j, r in enumerate(rotated_images):
        axs[i, j].imshow(r)
        axs[i, j].axis('off')



