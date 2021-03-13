# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:34:15 2021

@author: DELL E6430
"""

#importamos las librerias necesarias
import cv2;
import numpy as np;

#probamos que el xml este funcionando, lo cargamos
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#cargamos la imagen y la transformamos 
image = cv2.imread('oficina.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#debemos aplicar el clasificador en la imagen
#primer parametro -->imagen sobre la cual va a actuar
#scale factor --> que tanto se reduce la imagen, debemos jugar con los valores, ya que puede variar el reconocimiento
#minNeighbors -->todos los rectangulos o detecciones de un mismo rostro
#minSize -->El tamaño minimo posible del objeto
#maxSize--> El tamaño maximo posible del objeto
faces = faceClassif.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30,30),
                                     maxSize=(200,200))
#si se detecta rostro guardamos los puntos x y alto y ancho
for (x,y,w,h) in faces:
  cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

#visualizacion
cv2.imshow('image', image);
cv2.waitKey(0);
cv2.destroyAllWindows();