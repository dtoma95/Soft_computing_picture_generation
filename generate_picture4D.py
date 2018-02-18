
# Tomislav Dobricki SW21/2014 FTN - Novi Sad


# import the neural network
from digits_CNN.model import *

# import the necessary packages

from keras.models import Sequential
from keras.layers import Activation

from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from keras.callbacks import ModelCheckpoint

def save_img(data, path):
    if(RGB == 3):
        new_image = np.ones((HEIGHT,WIDTH,RGB), np.uint8)
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for k in range(0, RGB):
                    new_image[i][j][k] = data[i][j][k]*255
        cv2.imwrite(path, new_image)
    else:
        new_image = np.ones((HEIGHT,WIDTH,3), np.uint8)
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for k in range(0, 3):
                    new_image[i][j][k] = data[i][j][0]*255
        cv2.imwrite(path, new_image)

def genetic(data, current_maxes, model):
    for j in range(0, WIDTH):
        for i in range(0, HEIGHT):
            for k in range(0, RGB):
                data[i][i][j][k] = 0
        lista = model.predict(data,batch_size=HEIGHT,verbose=0)
        for i in range(0,HEIGHT):
            
            if lista[i][CLASS] > current_maxes[i]:
                print("BETTER")
                current_maxes[i] = lista[i][CLASS]
            else:
                print("WORSE")
                data[i][i][j][k] = 1
        print(str(j) + ". iteration over")
    return data

HEIGHT = 28
WIDTH = 28
RGB = 1
    
def image_to_feature_vector(image, size=(HEIGHT, WIDTH)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

CLASS = 9

if __name__ == '__main__':
    model = getInstance()
    HEIGHT, WIDTH, RGB = dimensions()
    
    blank_image = np.ones((HEIGHT,WIDTH,RGB), np.uint8)
    data = []
    features = blank_image
    data.append(features)
            
    data = np.array(data) #/ 255.0
    lista = model.predict(data,batch_size=1,verbose=0)
    print (lista)
    basic_value = lista[0][CLASS]

    data = []
    current_maxes = []
    for i in range(0, HEIGHT):
        data.append(np.ones((HEIGHT,WIDTH,RGB), np.uint8))
        current_maxes.append(basic_value)

    data = np.array(data)
        
    retval = genetic(data, current_maxes, model)    
    result = np.ones((HEIGHT,WIDTH,RGB), np.uint8)
    for i in range(0, HEIGHT):
        result[i] = data[i][i]
    
    print(result)
    
    data = []
    data.append(result)
    data = np.array(data)
    lista = model.predict(data,batch_size=1,verbose=0)
    print (lista)
    print (lista[0][CLASS])
    
    save_img(result, "result" + str(CLASS) +".png")