from multiprocessing.spawn import prepare
from tabnanny import verbose
import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
from keras.models import load_model
import numpy as np
import os
import PIL.Image, PIL.ImageTk
from cv2 import WND_PROP_VISIBLE


class App:

    def __init__(self):      

        cam = cv.VideoCapture(0)
        
        CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

        classif = cv.CascadeClassifier('other_data/haarcascade_frontalface_default.xml')
        filepath = './saved_model'
        model = load_model(filepath, compile = True)
                
        while True:
            
            #take frame
            ref, frame = cam.read()
            cv.imshow('cam', frame)
            
            key = cv.waitKey(1)
            
            #close window
            if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
                break
        
                    
            #change 
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = classif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            
            for x,y,w,h in faces:
                face = gray[y: y+h, x: x+w]
                
                if len(faces) != 0:
                    
                    prediction = model.predict([prepare('frame.jpg')])
                    print(prediction)   
            
        cam.release()
        
        
    def prepare(filepath):

        IMG_SIZE = 38
        
        img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        img_array = img_array.resize(IMG_SIZE, IMG_SIZE)
        
        new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)            
    
