from keras.models import Sequential
from keras.layers import Activation,PReLU,LeakyReLU
from keras.layers import Conv2D
from argparse import ArgumentParser
from keras.optimizers import SGD, Adam
from preprocess_data import get_data
import math
from keras import backend as K

#ui;d the model architecture,it contains 3 conv layers that connects the input image to the output image
def build_model(input_size):
    input_shape = (input_size[0], input_size[1],3)
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(9, 9), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('linear'))
    return model

if __name__=="__main__":
 X,Y=get_data()   #get the X and Y foro training from the preprocess_data.py
 print(X.shape,Y.shape)
 model = build_model((100,100))
 optimizer = Adam(lr=0.001)
 model.compile(loss='mse', optimizer=optimizer)

 model.fit(X, Y, batch_size=40,validation_split=0.12, epochs=30)
 model.save_weights("model.h5")     #save the model in a h5 file



