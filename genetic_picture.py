
# Tomislav Dobricki SW21/2014 FTN - Novi Sad


# import the neural network
from Landmarks_nn.model import *

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
        iterator = 0
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for k in range(0, RGB):
                    new_image[i][j][k] = data[iterator]*255
                    iterator = iterator + 1
        cv2.imwrite(path, new_image)
    else:
        new_image = np.ones((HEIGHT,WIDTH,3), np.uint8)
        iterator = 0
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for k in range(0, 3):
                    new_image[i][j][k] = data[iterator]*255
                iterator = iterator + 1
        cv2.imwrite(path, new_image)

def genetic(data, current_maxes, model):
    print("MAXES")
    print(current_maxes)
    for j in range(0, WIDTH*RGB):
        for i in range(0, HEIGHT):
            data[i][i*WIDTH*RGB+j] = 1
        lista = model.predict(data,batch_size=HEIGHT,verbose=0)
        for i in range(0,HEIGHT):
            print(lista[i])
            if lista[i][CLASS] > current_maxes[i]:
                print("BETTER")
                current_maxes[i] = lista[i][CLASS]
                if current_maxes[i] > 0.9:
                    print(i*WIDTH*RGB+j)
                    print(i, j)
                    #save_img(data[i], "kurcina")
            else:
                print("WORSE")
                data[i][i*WIDTH*RGB+j] = 0
        print(str(j) + ". iteration over")
    
    return data

HEIGHT = 105
WIDTH = 32
RGB = 1
    
def image_to_feature_vector(image, size=(HEIGHT, WIDTH)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

CLASS = 2

if __name__ == '__main__':
    model = getInstance()
    HEIGHT, WIDTH, RGB = dimensions()
    
    blank_image = np.zeros((HEIGHT,WIDTH,RGB), np.uint8)
    data = []
    features = image_to_feature_vector(blank_image, (HEIGHT, WIDTH))
    data.append(features)
            
    data = np.array(data) #/ 255.0
    lista = model.predict(data,batch_size=1,verbose=0)
    print (lista)
    basic_value = lista[0][CLASS]

    data = []
    current_maxes = []
    for i in range(0, HEIGHT):
        data.append(image_to_feature_vector(np.zeros((HEIGHT,WIDTH,RGB), np.uint8), (HEIGHT, WIDTH)))
        current_maxes.append(basic_value)

    data = np.array(data)
        
    retval = genetic(data, current_maxes, model)    
    result = []
    for red in data:
        
        if(result == []):
            result = red
        else:
            result = red + result
    #print(result)
    #result = result - (HEIGHT - 1)
    
    data = []
    data.append(result)
    data = np.array(data)
    lista = model.predict(data,batch_size=1,verbose=0)
    print (lista)
    print (lista[0][CLASS])
    
    save_img(result, "result" + str(CLASS) +".png")

