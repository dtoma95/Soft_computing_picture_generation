
# Tomislav Dobricki SW21/2014 FTN - Novi Sad


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

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=3072, kernel_initializer="uniform",
	activation="relu"))
model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] loading weights...")
model.load_weights('end_result.h5')

	
	
data = []
imagePath = "done.png"
image = cv2.imread(imagePath)
features = image_to_feature_vector(image)
data.append(features)
		
data = np.array(data) / 255.0
lista = model.predict(data,batch_size=1,verbose=0)
print "[Validation] Input of 20 pictures of faces"
print lista
for i in lista:
	if i[0] < i[1]:
		print "NOT_FACE"
	else:
		print "FACE" 