from keras.models import Sequential,Model
from keras.layers import Activation,Add
from keras.layers import Conv2D,BatchNormalization,LeakyReLU,Input
from argparse import ArgumentParser
from keras.optimizers import  Adam
import cv2
import numpy as np
import glob
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
import math

TRAINING_FOLDER="data/train/General-100/"

scale=2
def batch_generator(batch_size):
    while True:
        for img in glob.glob(TRAINING_FOLDER + "*.bmp"):
            batch_counter = 0
            imageY = cv2.imread(img)
            scale_factor = 2
            imageY = cv2.resize(imageY, (imageY.shape[0] // scale_factor, imageY.shape[1] // scale_factor))
            Xbatch = np.zeros((batch_size, imageY.shape[0], imageY.shape[1], 3))
            Ybatch = np.zeros((batch_size, imageY.shape[0], imageY.shape[1], 3))
            while batch_counter < batch_size:
                flipped = cv2.flip(imageY, np.random.randint(-1, 2))
                Ybatch[batch_counter, :, :, :] = flipped / 255
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(30, 90)]
                imag, result = cv2.imencode('.jpg', flipped, encode_param)
                imageX = cv2.imdecode(result, 1)

                Xbatch[batch_counter, :, :, :] = imageX / 255
                batch_counter += 1
            yield Xbatch, Ybatch
            batch_counter = 0


#ui;d the model architecture,it contains 3 conv layers that connects the input image to the output image
def build_model(scale):
    input_layer = Input(shape = (None,None,1))
    conv0 = Conv2D(32, (3,3), activation = 'tanh',padding = "SAME")(input_layer)

    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(conv0)
    conv2 = Conv2D(32, (3,3), activation = 'linear', padding = "SAME")(conv1)
    res0=Add()([conv0,conv2])

  
    conv3 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(res0)
    conv4 = Conv2D(32, (3,3), activation = 'linear', padding = "SAME")(conv3)
    res1=Add()([conv0,conv2,conv4])

    conv5 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(res1)
    conv6 = Conv2D(32, (3,3), activation = 'linear', padding = "SAME")(conv5)
    res2=Add()([conv0,conv4,conv6])
    """
    conv7 = Conv2D(256, (3,3), activation = 'relu', padding = "SAME")(res2)
    conv8 = Conv2D(256, (3,3), activation = 'linear', padding = "SAME")(conv7)
    res3=Add()([conv0,conv6,conv8])

    conv9 = Conv2D(256, (3,3), activation = 'relu', padding = "SAME")(res3)
    conv10 = Conv2D(256, (3,3), activation = 'linear', padding = "SAME")(conv9)
    res4=Add()([conv0,conv8,conv10])

    conv11 = Conv2D(256, (3,3), activation = 'relu', padding = "SAME")(res4)
    conv12 = Conv2D(256, (3,3), activation = 'linear', padding = "SAME")(conv11)
    res5=Add()([conv0,conv10,conv12])"""


    conv13 = Conv2D(16, (3,3), activation = 'relu', padding = "SAME")(res2)
    conv14 = Conv2D(8, (3,3), activation = 'relu', padding = "SAME")(conv13)
    
    
    
    conv15 = Conv2D(1, (3,3), activation = 'relu', padding = "SAME")(conv14)
    model = Model(input_layer,conv15)
    return model

def customLoss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=255, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)
    dssim = pow(((1 - ssim) / 2), 2)
    return dssim



if __name__=="__main__":
    model = build_model()
    optimizer = Adam(lr=0.001)
    callbacks = [ModelCheckpoint(filepath='model.h5', monitor='loss', save_best_only=True)]

    model.compile(loss=customLoss, optimizer=optimizer)
    history = model.fit_generator(batch_generator(2),
                                  callbacks=callbacks,
                                  samples_per_epoch=191, nb_epoch=2)







