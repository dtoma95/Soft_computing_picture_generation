
from keras.models import Sequential
from keras.layers import Activation

from keras.layers import Dense
from keras.utils import np_utils

def getInstance():
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(768, input_dim=3072, kernel_initializer="uniform",
        activation="relu"))
    model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    
    print("[INFO] loading weights...")
    model.load_weights('face_detection_NN/end_result.h5')
    return model
    
def dimensions():
    return 32, 32, 3