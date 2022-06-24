from tabnanny import verbose
import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
from keras.models import Sequential, save_model, load_model
import numpy as np
import os
import PIL.Image, PIL.ImageTk
from cv2 import WND_PROP_VISIBLE

class App:

    def __init__(self):      

        cam = cv.VideoCapture(0)
        
        IMG_SIZE = 38
        pred = []
        classif = cv.CascadeClassifier('other_data/haarcascade_frontalface_default.xml')
        filepath = './saved_model'
        model = load_model(filepath, compile = True)
                
        i = 0
        while True:
            i+=1
            
            ref, frame = cam.read()
            
            cv.imshow('cam', frame)
            
            key = cv.waitKey(1)
            
            if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
                break
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = classif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            
            for x,y,w,h in faces:
                face = gray[y: y+h, x: x+w]
                
                if len(faces) != 0:
                    
                    pred = np.array(pred, face)
                    pred = np.array(pred).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

                    predicts = model.predict(x=pred, batch_size=10, verbose=0)
                    
                    for aux in predicts:
                        print(aux)

                    
                    """ prediction_img = []
                    prediction_img[0].reshape((48, 48))
                    
                    sample_to_predict = []
                    sample_to_predict.append(prediction_img[0]) """
                    
                    
                    
                    print(i)
                else:
                    print(-i)
                
                if face.shape[0] >= 200 and face.shape[1] >= 200:
                    face = cv.resize(face, (48, 48))
                    
            

            
            
        cam.release()
        
        


"""     def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update) """

"""     def predict(self):
        CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        if prediction == 1:
            self.class_label.config(text=CATEGORIES[0])
            return CATEGORIES[0]
        
        if prediction == 2:
            self.class_label.config(text=CATEGORIES[1])
            return CATEGORIES[1]
        
        if prediction == 3:
            self.class_label.config(text=CATEGORIES[2])
            return CATEGORIES[2]
        
        if prediction == 4:
            self.class_label.config(text=CATEGORIES[3])
            return CATEGORIES[3]
        
        if prediction == 5:
            self.class_label.config(text=CATEGORIES[4])
            return CATEGORIES[4]
        
        if prediction == 6:
            self.class_label.config(text=CATEGORIES[5])
            return CATEGORIES[5]
        
        if prediction == 7:
            self.class_label.config(text=CATEGORIES[6])
            return CATEGORIES[6] """