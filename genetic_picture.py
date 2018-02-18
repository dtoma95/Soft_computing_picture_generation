
# Tomislav Dobricki SW21/2014 FTN - Novi Sad


# import the neural network
from digits_NN/model import *

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
    new_image = np.ones((HEIGHT,WIDTH,3), np.uint8)
    iterator = 0
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            for k in range(0, 3):
                new_image[i][j][k] = data[iterator]*255
                iterator = iterator + 1

    print ("Savin")
    cv2.imwrite(path, new_image)

def genetic(data, current_maxes, model):
    #print("MAXES")
    #print(current_maxes)
    for j in range(0, WIDTH*3):
        for i in range(0, HEIGHT):
            data[i][i*WIDTH*3+j] = 0
        lista = model.predict(data,batch_size=HEIGHT,verbose=0)
        for i in range(0,HEIGHT):
            
            if lista[i][0] > current_maxes[i]:
                print("BETTER")
                current_maxes[i] = lista[i][0]
                if current_maxes[i] > 0.9:
                    print(i*WIDTH*3+j)
                    print(i, j)
                    #save_img(data[i], "kurcina")
            else:
                print("WORSE")
                data[i][i*WIDTH*3+j] = 1
        print(str(j) + ". iteration over")
    print current_maxes
    return data

def image_to_feature_vector(image, size=(HEIGHT, WIDTH)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


HEIGHT = 32
WIDTH = 32

if __name__ == '__main__':
    model = getInstance()
    HEIGHT, WIDTH = dimensions()
    
    blank_image = np.ones((HEIGHT,WIDTH,3), np.uint8)
    data = []
    features = image_to_feature_vector(blank_image)
    data.append(features)
            
    data = np.array(data) #/ 255.0
    lista = model.predict(data,batch_size=1,verbose=0)
    print (lista)
    basic_value = lista[0][0]

    data = []
    current_maxes = []
    for i in range(0, HEIGHT):
        data.append(image_to_feature_vector(np.ones((HEIGHT,WIDTH,3), np.uint8)))
        current_maxes.append(basic_value)

    data = np.array(data)
        
    retval = genetic(data, current_maxes, model)    
    result = []
    for red in data:
        if(result == []):
            result = red
        else:
            result = red + result

    result = result - HEIGHT - 1
            
    save_img(result, "result.png")

