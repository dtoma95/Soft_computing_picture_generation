import cv2
import numpy as np
from skimage.feature import hog
from model import *

from settings import *

def get_features(image):
    """
    Function extracts features from a list of images

    :param images: list of images
    :return: list of hog features for each image from the input list
    """

    image_features = []
    image = cv2.resize(image, image_size)

    # Color features
    if color_histogram:
        hist_features = color_hist(image)
        image_features.append(hist_features)

    # HOG descriptor
    if hog_descriptor:
        hog_features = []
        if all_channels:
            for channel in range(image.shape[2]):
                hog_features.append(get_hog_features(image[:, :, channel], visualise=False))
            hog_features = np.ravel(hog_features)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features = get_hog_features(image, visualise=False)
        image_features.append(hog_features)
    return np.concatenate(image_features)

def get_hog_features(image, visualise=False):
    """
    Function calculates HOG features and visualization (optionally)
    :param image: input image
    :param visualise: flag which indicates whether the hog image should also be returned
    :return: HOG features and visualisation (optionally)
    """
    hog_features, hog_image = hog(image, orientations=orientations,
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block), block_norm='L1',
                                  transform_sqrt=transform_sqrt, visualise=True, feature_vector=feature_vector)
    print(len(hog_features))
    return hog_features, hog_image



image = cv2.imread("Big Ben.jpg")
r1, r2 = get_features(image)

val = np.zeros((105,32,1), np.uint8).flatten()

model = getInstance()
    
data = []
data.append(val)
            
data = np.array(data) #/ 255.0
lista = model.predict(data,batch_size=1,verbose=0)
print (lista)
