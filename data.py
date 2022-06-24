""" import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

import numpy as np
import train
import cv2 as cv
import os """
from ast import dump
from cProfile import label
from dbm import dumb
from importlib.resources import path
from sre_parse import CATEGORIES
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import glob
import random
import pickle

DATADIR = 'train/'
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_SIZE = 48
        
training_data = []
train = np.array([])
test = np.array([])
    
for category in CATEGORIES:
    path = glob.glob(DATADIR + category + '/*.png')
    class_num = CATEGORIES.index(category)
    #print(path)
            
    for img in path:
        try:
            img_array = cv.imread(img, cv.IMREAD_GRAYSCALE)
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            #print(os.path.join(path, img))
        except Exception as e:
            pass
                
#print(len(training_data))

random.shuffle(training_data)

print(len(training_data))
        
for sample in training_data[:10]:
    print(sample[1])
    
cont = 0

for features, label in training_data:  
    print(cont)
    cont += 1
    train = np.append(train, features)
    test = np.append(test, label) # arrunasr
            
    train = np.array(train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
print(train.shape)
print(test.shape)
            
pickle_out = open("train.pickle", "wb")
pickle.dump(train, pickle_out)
pickle_out.close()


pickle_out = open("test.pickle", "wb")
pickle.dump(test, pickle_out)
pickle_out.close()
            
        
"""         classif = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        x = []

        image_size = (48, 48)
        batch_size = 32

        for name in classif:
            
            data = []
            paths = 
            print(name)
            for path in paths:
                if path.endswith('.png'):
                    image = cv.imread (path)
                    data.append (image) 
                    
            x = [name , np.array(data)]


        x = np.array(x, dtype=2) """
