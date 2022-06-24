import pickle
from pyexpat import model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import cv2 as cv
import PIL

class Model:
    
    def __init__(self):
        train = pickle.load(open('train.pickle', 'rb'))
        test = pickle.load(open('test.pickle', 'rb'))
        
        #train = train/255.0
        
        model = Sequential()
        
        model.add(Conv2D(64, (3,3), input_shape = train.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(64, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(64))
        
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.fit(train, test, epochs=1, batch_size=32, validation_split=0.1)
        
"""     def predict(frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread('frame.jpg')[:, :, 0]
        img = img.reshape(16800)
        prediction = model.predict([img])

        return prediction[0] """