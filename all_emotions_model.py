import numpy as np
import cv2 as cv
import glob
import random
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, Activation

        
DATADIR = 'train/'
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_SIZE = 224
            
aux_emo = 0
        
#all_emotions
training_data = []
init_train = [] #np.array([])
init_test = [] #np.array([])
train = None             
                    
training_data = []
init_train = [] #np.array([])
init_test = [] #np.array([]) 
            
for emotion in CATEGORIES:
    path = glob.glob(DATADIR + emotion + '/*.png')
    class_num = CATEGORIES.index(CATEGORIES[aux_emo])
    
    for img in path:
        try:
            img_array = cv.imread(img)#, cv.COLOR_BGR2RGB)
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, aux_emo])
            #print(os.path.join(path, img))
        except Exception as e:
            pass
    aux_emo += 1
                    
print(len(training_data))

random.shuffle(training_data)
                
print(len(training_data))
                            
for sample in training_data[:10]:
    print(sample[1])
                        
cont = 0

for features, label in training_data:  
    print(str(cont) + '/' + str(len(training_data)))
    cont += 1
    init_train.append(features)
    init_test.append(label)
                                
    train = np.array(init_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)           
                    
#print(train.shape)
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
                    
new_model.fit(train, test, epochs = 25)
                        
#filepath = './saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo]
#save_model(new_model, filepath)
                    
new_model.save('./saved_model/all_emotions.h5')
