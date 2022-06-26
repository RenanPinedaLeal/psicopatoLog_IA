from ast import dump
from cProfile import label
from dbm import dumb
from importlib.resources import path
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import glob
import random
from pyexpat import model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import PIL

class Data:
    
    #def __init__(self):
        
    DATADIR = 'train/'
    CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    IMG_SIZE = 224
        
    aux_emo = 0
    aux_already_gone = 2
                

        
    while aux_emo == 0:
        #print(CATEGORIES[aux_emo])
                
        aux_secemo = aux_already_gone
            
        while aux_secemo == 2:
            if(aux_secemo != aux_emo):
                
                print(CATEGORIES[aux_emo] + ' and ' + CATEGORIES[aux_secemo])
                
                training_data = []
                train = np.array([])
                test = np.array([]) 
                
                path = glob.glob(DATADIR + CATEGORIES[aux_emo] + '/*.png')
                class_num = CATEGORIES.index(CATEGORIES[aux_emo])
                
                path_sec = glob.glob(DATADIR + CATEGORIES[aux_secemo] + '/*.png')
                class_num_sec = CATEGORIES.index(CATEGORIES[aux_secemo])
                                
                for img in path:
                    try:
                        img_array = cv.imread(img, cv.IMREAD_GRAYSCALE)
                        new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        training_data.append([new_array, class_num])
                        #print(os.path.join(path, img))
                    except Exception as e:
                        pass
                    
                for img in path_sec:
                    try:
                        img_array = cv.imread(img, cv.IMREAD_GRAYSCALE)
                        new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        training_data.append([new_array, class_num_sec])
                        #print(os.path.join(path, img))
                    except Exception as e:
                        pass
                    
                #print(len(training_data))

                random.shuffle(training_data)

                #print(len(training_data))
                        
                #for sample in training_data[:10]:
                    #print(sample[1])
                    
                cont = 0

                for features, label in training_data:  
                    print(str(cont) + '/' + str(len(training_data)))
                    cont += 1
                    train = np.append(train, features)
                    test = np.append(test, label) # arrunasr
                            
                    train = np.array(train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)           
                
                #print(train.shape)
                #print(test.shape)
                
                train = train/255.0
                
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
                
                new_model.fit(train, test, epochs = 25)
                    
                """ model = Sequential()
                    
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
                    
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                    
                model.fit(train, test, epochs=5, batch_size=32, validation_split=0.1) """

                filepath = './saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo]
                save_model(model, filepath)
                
            aux_secemo += 1
            
        aux_already_gone += 1

        aux_emo += 1
            
        """ for category in CATEGORIES:
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
            print(cont + '/' + len(training_data))
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
        pickle_out.close() """
            
