import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import time

seed = 3
np.random.seed(seed)

def function():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print (x_train.shape)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(384, kernel_initializer="uniform", activation="relu"))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    startTime = time.time()
    print(startTime)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print ("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    endTime = time.time()
    print(endTime)
    print(startTime - endTime)
    model.save_weights('end_result_new.h5')
    
if __name__ == '__main__':
    function()