
from keras.models import Sequential
from keras.layers import Activation

from keras.layers import Dense
from keras.utils import np_utils

def getInstance():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=3360))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    
    print("[INFO] loading weights...")
    model.load_weights('Landmarks_nn/model.h5')
    return model
    
def dimensions():
    return 105, 32, 1
    
    
    #landmarks = ["Big Ben", "Burj Al Arab", "Eiffel Tower", "Golden Gate Bridge", "Leaning Tower of Pisa",
             #"Manneken Pis", "nonlandmark", "Statue of Liberty"]