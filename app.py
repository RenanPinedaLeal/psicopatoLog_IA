from multiprocessing.spawn import prepare
from sre_parse import CATEGORIES
from tabnanny import check, verbose
import tkinter as tk
from tkinter import simpledialog
from turtle import shape
from keras.models import load_model

import os
import PIL.Image, PIL.ImageTk
from cv2 import WND_PROP_VISIBLE, cvtColor

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from keras.applications.mobilenet_v2 import preprocess_input
import cv2 as cv


class App:
    
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
   

    def __init__(self):      

        cam = cv.VideoCapture(0)

        i = 0
        while True:
            i+=1

            #take frame
            ref, frame = cam.read()

            if frame.any() == None:
                return

            cv.imshow('cam', frame)
            
            self.pred(frame = frame)

            key = cv.waitKey(1)

            #close window
            if cv.getWindowProperty('cam', WND_PROP_VISIBLE) == 0:
                print('-------------------')
                #print(np.argmax(self.cont))
                print('most predominant emotion: ' + self.CATEGORIES[np.argmax(np.array(self.cont))])
                
                aux_cont = 0
                for category in self.CATEGORIES:
                    print(category + ': ' + str(self.cont[aux_cont]))
                    aux_cont += 1
                
                break
        
                
    def pred(self, frame ):

        #frame = cv.imread("frame2.jpg")
        #print(frame.shape)
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #print(gray.shape)

        faces = self.face_finder.detectMultiScale(gray, 1.1, 4)
        
        for x,y,w,h in faces:
            roi_gray = gray[y: y+h, x: x+w]
            #print(roi_gray.shape)
            
            roi_img = cv.cvtColor(roi_gray, cv.COLOR_BGR2RGB)
            #print(roi_img.shape)

            if(len(roi_img) == 0):
                return
            
            
            
            final_img = cv.resize(roi_img, (224, 224))
            final_img = np.expand_dims(final_img, axis=0)
            
            final_img = final_img/255.0
            
            pred = self.model.predict(final_img)
            
            #print(pred[0])
            print(self.CATEGORIES[np.argmax(pred)])
            self.cont[np.argmax(pred)] += 1