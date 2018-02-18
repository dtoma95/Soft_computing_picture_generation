
from keras.models import Sequential
from keras.layers import Activation

from keras.layers import Dense
from keras.utils import np_utils

def getInstance():
    model = Sequential()
    model.add(Dense(28L*28L, input_dim=28L*28L, kernel_initializer='normal', activation='relu'))
    model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    print("[INFO] loading weights...")
    model.load_weights('digits_NN/end_result_new.h5')
    return model
    
def dimensions():
    return 28, 28, 1