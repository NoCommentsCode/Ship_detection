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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

step = 500
scale_percent = 20 # percent of original size
W = 768
H = 768
    
def cnn_predict(model, image_file):
    img = cv2.imread(image_file)
    
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)   
    
    image_size = img.shape[0]    
    img_arr = np.expand_dims(img, axis=0)
    img_arr = img_arr.astype("float32")/255.0 #1 - img_arr/255.0
    img_arr = img_arr.reshape((1, image_size, image_size, 3))
    
    result = model.predict(img_arr)
    return np.argmax(result)


def check(csv_dir, image_folder):
    names = []
    with open(csv_dir) as File:
        data = csv.reader(File)
        for i in data:
            names.append(i)   
            
    names.pop(0)
    n = 0
    t = 0
    for i in range(len(names)):
        n+=1
        img = cv2.imread(os.path.join(image_folder, names[i][0])) 
        a = cnn_predict(model, os.path.join(image_folder, names[i][0]))
        if(a==int(names[i][1])):
            t+=1
        print(names[i][0], ": ", a==int(names[i][1]))
        
    print("Accuracy: ", t*100/n, "%")
        
        
def output(model):
    db1 = pd.DataFrame(columns = ["ImageId", "IdCls"])
    for fname in os.listdir("test/test"):
        db1.loc[db1.shape[0]] = {'ImageId':fname, 'IdCls': cnn_predict(model, os.path.join("test/test", fname))}
    db1.to_csv('output.csv', index = False)
    
    
model = tf.keras.models.load_model('ship_class.h5')