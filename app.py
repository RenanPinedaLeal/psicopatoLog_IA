import numpy as np
from keras.models import load_model
from keras.models import load_model
import cv2 as cv
from cv2 import WND_PROP_VISIBLE

    
IMG_SIZE = 244
CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
cont = [0, 0, 0, 0, 0, 0, 0]
face_finder = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
aux_emo = 0
aux_secemo = 1

m_01 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[1] + '.h5')
m_02 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[2] + '.h5')
m_03 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[3] + '.h5')
m_04 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[4] + '.h5')
m_05 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[5] + '.h5')
m_06 = load_model('./saved_model/' + CATEGORIES[0] + '_' + CATEGORIES[6] + '.h5')

m_12 = load_model('./saved_model/' + CATEGORIES[1] + '_' + CATEGORIES[2] + '.h5')
m_13 = load_model('./saved_model/' + CATEGORIES[1] + '_' + CATEGORIES[3] + '.h5')

#print(filepath)
        
#model = load_model('./saved_model/all_emotions.h5')
   
cam = cv.VideoCapture(0)

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def predict():
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_finder.detectMultiScale(gray, 1.1, 4)
            
    for x,y,w,h in faces:
        roi_gray = gray[y: y+h, x: x+w]
                
        roi_img = cv.cvtColor(roi_gray, cv.COLOR_BGR2RGB)

        if(len(roi_img) == 0):
            pass
        
        #cv.imshow('cam', roi_img)
                
        final_img = cv.resize(roi_img, (224, 224))
        final_img = np.expand_dims(final_img, axis=0)
                
        final_img = final_img/255.0
        
        pred = [np.argmax(m_01.predict(final_img)), 
                            np.argmax(m_02.predict(final_img)), 
                            np.argmax(m_03.predict(final_img)), 
                            np.argmax(m_04.predict(final_img)),
                            np.argmax(m_05.predict(final_img)),
                            np.argmax(m_06.predict(final_img)),
                            np.argmax(m_12.predict(final_img)),
                            np.argmax(m_13.predict(final_img)),]

        final_pred = most_frequent(pred)
        
        print(CATEGORIES[final_pred])
        
        print(final_pred)
        
        print(str(pred[0]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[1])
        print(str(pred[1]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[2])
        print(str(pred[2]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[3])
        print(str(pred[3]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[4]) 
        print(str(pred[4]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[5]) 
        print(str(pred[5]) + ' | ' + CATEGORIES[0] + '_' + CATEGORIES[6]) 
        
        print(str(pred[6]) + ' | ' + CATEGORIES[1] + '_' + CATEGORIES[2]) 
        print(str(pred[7]) + ' | ' + CATEGORIES[1] + '_' + CATEGORIES[3]) 

        cont[final_pred] += 1

i = 0
while True:
    i+=1

    #take frame
    ref, frame = cam.read()

    if frame.any() == None:
        break

    cv.imshow('cam', frame)
    
    #predict            
    predict()
    #finished predict    
    
    key = cv.waitKey(1)

    #close window
    if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
        print('-------------------')
        print('most predominant emotion: ' + CATEGORIES[np.argmax(np.array(cont))])
            
        aux_cont = 0
        for category in CATEGORIES:
            print(category + ': ' + str(cont[aux_cont]))
            aux_cont += 1
                
        break