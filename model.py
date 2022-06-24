import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

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
        
        model.fit(train, test, epochs=2, batch_size=32, validation_split=0.1)