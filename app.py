import numpy as np
from keras.models import load_model
from keras.models import load_model
import cv2 as cv
from cv2 import WND_PROP_VISIBLE
import os
import time

    
IMG_SIZE = 244
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

cont = [0, 0, 0, 0, 0, 0, 0]
face_finder = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


models = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None ]
aux_mod = 0
ansr_final = [0, 0]

for category in CATEGORIES:
    for category2 in CATEGORIES:
        if(os.path.exists('./saved_model/' + category + '_' + category2 + '.h5')):
            print('--FOUND ' + str(aux_mod) + '--')
            models[aux_mod] = load_model('./saved_model/' + category + '_' + category2 + '.h5')
            aux_mod += 1
            
INIT_TIME = time.time()
cam = cv.VideoCapture(0)

def most_frequent(List):
    aux_ele = [0, 0, 0, 0, 0, 0, 0]
    limit = 5
    
    for ele in List:
        if ele == 0:
            aux_ele[0] += 1
        elif ele == 1:
            aux_ele[1] += 1
        elif ele == 2:
            aux_ele[2] += 1
        elif ele == 3:
            aux_ele[3] += 1
        elif ele == 4:
            aux_ele[4] += 1
        elif ele == 5:
            aux_ele[5] += 1
        elif ele == 6:
            aux_ele[6] += 1
            
    #os.system('cls')
        
    for e in aux_ele:
        print(e)    
    
    if aux_ele[1] == 6:
        return False
    elif aux_ele[3] == 6:
        return False
    elif aux_ele[4] == 6:
        return False
    elif aux_ele[6] == 6:
        return False    
    
    if aux_ele[0] >= limit:
        return True
    elif aux_ele[2] >= limit:
        return True
    elif aux_ele[5] >= limit:
        return True
    
    return False

def predict(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_finder.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:    
        print('--FACE NOT FOUND--')
        
    
    for x,y,w,h in faces:
        roi_gray = gray[y: y+h, x: x+w]
                
        roi_img = cv.cvtColor(roi_gray, cv.COLOR_BGR2RGB)

        if(len(roi_img) == 0):
            pass
                        
        final_img = cv.resize(roi_img, (224, 224))
        final_img = np.expand_dims(final_img, axis=0)
                
        final_img = final_img/255.0
        
        pred = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None ]
        aux_pred = 0
        
        for mod in models:
            if (mod != None):
                pred[aux_pred] = np.argmax(mod.predict(final_img))
                #os.system('cls')

                aux_pred += 1
            else:
                print('--IS EMPTY--')
        
        
        final_pred = most_frequent(pred)
        return final_pred
        
def main(how_much):
    i = 0
    while True:
        i+=1

        #take frame
        ref, frame = cam.read()

        if frame.any() == None:
            break

        cv.imshow('cam', frame)
        
        #predict            
        ansr = predict(frame)
        
        if ansr != None:
            print(str(ansr))   
            if ansr == True:
                ansr_final[0] += 1
            else:
                ansr_final[1] += 1
        #finished predict    
        
        key = cv.waitKey(1)

        #close window        
        if (cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0) or (time.time() - INIT_TIME > how_much):   
            print('--------------')
            print('True: ' + str(ansr_final[0]))             
            print('False: ' + str(ansr_final[1]))             
            print('--------------')
            
            if (ansr_final[0] == 0):
                print(False)        
            elif ansr_final[1]/ansr_final[0] < 1.5:
                print(True)
            else:
                print(False)
            
            break
        
main(20)