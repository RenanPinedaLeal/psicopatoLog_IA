import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import glob
import random
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import PIL

        
DATADIR = 'train/'
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_SIZE = 224
            
aux_emo = 3
aux_already_gone = aux_emo
        
#all_emotions
training_data = []
init_train = [] #np.array([])
init_test = [] #np.array([])             
            
#pair of emotions    
while aux_emo <= 6:
    #print(CATEGORIES[aux_emo])
    if (aux_emo == 3):
        aux_secemo = aux_already_gone + 2
    else:        
        aux_secemo = aux_already_gone
                
    while aux_secemo <= 6:
        if(aux_secemo != aux_emo):
                    
            print(CATEGORIES[aux_emo] + ' and ' + CATEGORIES[aux_secemo])
                    
            training_data = []
            init_train = [] #np.array([])
            init_test = [] #np.array([]) 
                    
            path = glob.glob(DATADIR + CATEGORIES[aux_emo] + '/*.png')
            class_num = CATEGORIES.index(CATEGORIES[aux_emo])
                    
            path_sec = glob.glob(DATADIR + CATEGORIES[aux_secemo] + '/*.png')
            class_num_sec = CATEGORIES.index(CATEGORIES[aux_secemo])
                                    
            for img in path:
                try:
                    img_array = cv.imread(img)#, cv.COLOR_BGR2RGB)
                    new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                    #print(os.path.join(path, img))
                except Exception as e:
                    pass
                        
            for img in path_sec:
                try:
                    img_array = cv.imread(img)#, cv.COLOR_BGR2RGB)
                    new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num_sec])
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
                print(str(cont) + '/' + str(len(training_data)))
                cont += 1
                init_train.append(features)
                init_test.append(label) # arrunasr
                                
                train = np.array(init_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)           
                    
            print(train.shape)
            #print(np.array(test).shape)
                    
            train = train/255.0
            test = np.array(init_test)
                    
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
                    
            new_model.fit(train, test, epochs = 15)
                        
            #filepath = './saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo]
            #save_model(new_model, filepath)
                    
            new_model.save('./saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo] + '.h5')
                    
        aux_secemo += 1
                
    aux_already_gone += 1

    aux_emo += 1 
