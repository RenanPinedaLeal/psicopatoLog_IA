from multiprocessing.spawn import prepare
from tabnanny import verbose
import tkinter as tk
from tkinter import simpledialog
from turtle import shape
from keras.models import load_model

import os
import PIL.Image, PIL.ImageTk
from cv2 import WND_PROP_VISIBLE

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from keras.applications.mobilenet_v2 import preprocess_input
import cv2 as cv


class App:

    def __init__(self):      

        cam = cv.VideoCapture(0)
        
        IMG_SIZE = 244
        CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        face_finder = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        aux_emo = 0
        aux_secemo = 1
        filepath = './saved_model/' + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo] + '.h5'
        
        print(filepath)
        
        model = load_model(filepath) #load_model(filepath + CATEGORIES[aux_emo] + '_' + CATEGORIES[aux_secemo] + '/', compile = True)
        
        frame = cv.imread('frame.png')
        #print(frame.shape)
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #print(gray.shape)
            
        faces = face_finder.detectMultiScale(gray, 1.1, 4)
        #print(faces.shape)
        
        for x,y,w,h in faces:
            face = np.empty((1, 244, 244, 3))
            face[0] = np.array(gray).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  
            face = preprocess_input(face)

            if len(faces) != 0:
                
                """ data = np.empty((1, 244, 244, 3))
                data[0] = face
                data = preprocess_input(data) """
                """ mid_img = cv.resize(face, (244,244))
                final_img = np.array(final_img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)          
                final_img = np.expand_dims(final_img, axis=0)
                final_img = final_img/255.0 """
                
                
                pred = model.predict(face)
                print(np.argmax(pred))
                
            #fin_face = gray[y:y+h, x:x+w]
            #print(fin_gray.shape)
            
            #fin_color = frame[y:y+h, x:x+w]
            
            #cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            
            #mid_face = face_finder.detectMultiScale(fin_gray)
            #print(mid_face[0])
            
            """print(len(mid_face))
            
            if len(mid_face) == 0:
                print('--face not detected--')
                return
            else:  """
                           
            """ for ex,ey,ew,eh in mid_face:
                fin_face = fin_color[ey:ey+eh, ex:ex+ew] """
                #print(fin_face[0])
        
        
        
        #print(final_img)
        

            
        """ while True:
            
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
        
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)         """    
    
