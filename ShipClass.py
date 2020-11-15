from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import logging
import numpy as np
import time

import cv2
import csv
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scale_percent = 20 # percent of original size

# Initial image properties
W = 768
H = 768

# Loading list of test images names
def load_image_names(file):
    names = []
    with open(file) as File:
        data = csv.reader(File)
        for i in data:
            names.append(i)
        
    names.pop(0)
    random.shuffle(names)
    return names
 
# Partial image data loading (from name[begin] to name[end])
def load_data_pack(folder, names, begin, end):
    images = []
    labels = []
    for i in range(begin, end):
        img = cv2.imread(os.path.join(folder, names[i][0]))
    
        if img is not None:
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
            
            images.append(img)
            labels.append(names[i][2])
            
    return np.array(images), np.array(labels)
 
# List of images names
names = load_image_names('train_classification.csv')

# CNN model
def cnn_model():
    image_size = int(W * scale_percent / 100)
    num_channels = 3
    num_classes = 2
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training network
def cnn_train(model):
    # Main data properties
    image_size = int(W * scale_percent / 100)
    num_channels = 3
    num_classes = 2
    
    test_images, test_labels = load_data_pack("train", names, 20000,21000)
    val_data = np.reshape(test_images, (test_images.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)      
    
    print("Training the network...")
    for i in range(0, 20000, 500):
        train_images, train_labels = load_data_pack("train", names, i, i+500)
        
        train_data = np.reshape(train_images, (train_images.shape[0], image_size, image_size, num_channels))
        train_data = train_data.astype('float32') / 255.0
        
        train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)   
        
        model.fit(train_data, train_labels_cat, epochs=8, batch_size=64, validation_data=(val_data, val_labels_cat))
    
    return model


def cnn_predict(model, image_file):
    img = cv2.imread(image_file)
    
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)   
    
    image_size = img.shape[0]    
    img_arr = np.expand_dims(img, axis=0)
    img_arr = img_arr.astype("float32")/255.0
    img_arr = img_arr.reshape((1, image_size, image_size, 3))
    
    result = model.predict(img_arr)
    return result


model = cnn_model()
cnn_train(model)
model.save("ship_class.h5")