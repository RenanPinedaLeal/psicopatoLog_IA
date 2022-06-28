from keras.models import load_model

from cv2 import WND_PROP_VISIBLE

import numpy as np
from keras.models import load_model
import cv2 as cv


    
IMG_SIZE = 244
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
cont = [0, 0, 0, 0, 0, 0, 0]
face_finder = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
aux_emo = 0
aux_secemo = 1
filepath = './saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo] + '.h5'
        
#print(filepath)
        
#model = load_model(filepath)
model = load_model('./saved_model/all_emotions.h5')
   
cam = cv.VideoCapture(0)

i = 0
while True:
    i+=1

    #take frame
    ref, frame = cam.read()

    if frame.any() == None:
        break

    cv.imshow('cam', frame)
    
    #predict            
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #print(gray.shape)

    faces = face_finder.detectMultiScale(gray, 1.1, 4)
            
    for x,y,w,h in faces:
        roi_gray = gray[y: y+h, x: x+w]
        #print(roi_gray.shape)
                
        roi_img = cv.cvtColor(roi_gray, cv.COLOR_BGR2RGB)
        #print(roi_img.shape)

        if(len(roi_img) == 0):
            pass
                
        final_img = cv.resize(roi_img, (224, 224))
        final_img = np.expand_dims(final_img, axis=0)
                
        final_img = final_img/255.0
                
        pred = model.predict(final_img)
                
        #print(pred[0])
        print(CATEGORIES[np.argmax(pred)])
        cont[np.argmax(pred)] += 1
    #finished predict    
    
    key = cv.waitKey(1)

    #close window
    if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
        print('-------------------')
        #print(np.argmax(self.cont))
        print('most predominant emotion: ' + CATEGORIES[np.argmax(np.array(cont))])
            
        aux_cont = 0
        for category in CATEGORIES:
            print(category + ': ' + str(cont[aux_cont]))
            aux_cont += 1
                
        break