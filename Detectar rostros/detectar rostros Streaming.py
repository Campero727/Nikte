# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 17:25:01 2021

@author: DELL E6430
"""
#importamos las librerias necesarias
import cv2

#capturamos el video streaming con la web cam
cap = cv2.VideoCapture(0)

#cargamos el xml de opencv
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#realizamos la deteccion de rostros y el marcado del recatngulo
while True:
  ret,frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceClassif.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
  cv2.imshow('frame',frame)
  
  #comando para cerrar la aplicacion
  if cv2.waitKey(1) & 0xFF == ord('s'):
    break
cap.release()
cv2.destroyAllWindows()