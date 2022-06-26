import pickle
from pickletools import optimize
from pyexpat import model
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import cv2 as cv
import PIL


class Model:

    model = MobileNetV2()
                
    base_input = model.layers[0].input
    base_output = model.layers[-2].output

    final_output = Dense(128)(base_output)
    final_ouput = Activation('relu')(final_output)
    final_output = Dense(64)(final_ouput)
    final_ouput = Activation('relu')(final_output)
    final_output = Dense(7, activation='softmax')(final_ouput)
    
    new_model = keras.Model(inputs = base_input, outputs = final_output)
    
    new_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])