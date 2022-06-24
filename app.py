import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import cv2
from cv2 import WND_PROP_VISIBLE
import camera
import model

class App:

    def __init__(self):      

        cam = cv.VideoCapture(0)
        classif = cv.CascadeClassifier('other_data/haarcascade_frontalface_default.xml')
        
        while True:
            ref, frame = cam.read()
            
            cv.imshow('cam', frame)
            
            key = cv.waitKey(1)
            
            if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
                break
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = classif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            for x,y,w,h in faces:
                """ 
                face = gray[y: y+h, x: x+w]
                
                if face.shape[0] >= 200 and face.shape[1] >= 200:
                    face = cv.resize(face, (48, 48)) """
                    
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), thickness=3)
            
            
            
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