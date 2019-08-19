import cv2
import time
import numpy as np
import re
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

size = 2
total_sample_size = 10000



# download the xml file from here: https://github.com/Itseez/opencv/tree/master/data/haarcascades

def startCamera():
 face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')       #classifier to recognize face
 cap = cv2.VideoCapture(0)                                                         #capture webcam video
 while 1:
    ret, img = cap.read()         #read a frame
    cv2.resize(img, (500,400), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                  #convert to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                           #detect all the faces present on the image
    cv2.imshow('img', img)
    cv2.moveWindow('img',700,00)
    count = 0
    if len(faces)>0:                                                              #only if faces are found save the faces in the video

     #time.sleep(1.5)

     arrayOfFaces=[]
     for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)                #create a rectangle around image

        roi_color = img[y:y + h, x:x + w]
        #numpy_horizontal_concat = np.concatenate((img[y:y + h, x:x + w][1], gray[y:y + h, x:x + w]), axis=1)
        #cv2.destroyAllWindows()#
        dim = (200, 200)
        # resize image
        arrayOfFaces.append(cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA))
        #cv2.imshow('img'+str(count),resized  )
        count+=1
        #cv2.imwrite('found.png', roi_color)                                       #save the area of interest to memory after every 1.5 seconds
     if len(arrayOfFaces)==1:
         cv2.destroyWindow("img_multiple0")
         cv2.imshow('img' + str(0), arrayOfFaces[0])
         cv2.moveWindow('img' + str(0), 0,50);
     else:
         numpy_horizontal_concat = np.concatenate((arrayOfFaces[0], arrayOfFaces[1]), axis=0)
         for i in range(2,len(arrayOfFaces)):
              print('faces found',len(arrayOfFaces))
              cv2.destroyWindow("img0")
              numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, arrayOfFaces[i]), axis=0)
              cv2.imshow('img_multiple' + str(0), numpy_horizontal_concat)

              #time.sleep(3)
    else:
         while count>-1:
          cv2.destroyWindow("img"+str(count))
          count-=1
          #cv2.imshow('img', img)                                                   #else show the normal footage
    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

 cap.release()
 cv2.destroyAllWindows()




def inputMyFace():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # classifier to recognize face
    cap = cv2.VideoCapture(0)  # capture webcam video
    iteration=0
    while 1:
        iteration+=1
        ret, img = cap.read()  # read a frame
        cv2.resize(img, (500, 400), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # detect all the faces present on the image
        cv2.imshow('img', img)
        cv2.moveWindow('img', 700, 00)
        if len(faces) > 0:  # only if faces are found save the faces in the video

            # time.sleep(1.5)
            count = 0

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # create a rectangle around image

                roi_color = img[y:y + h, x:x + w]
                # numpy_horizontal_concat = np.concatenate((img[y:y + h, x:x + w][1], gray[y:y + h, x:x + w]), axis=1)
                # cv2.destroyAllWindows()#

                dim = (200,200)
                # resize image
                resized=cv2.resize(roi_color, dim, interpolation=cv2.INTER_AREA)
                cv2.imshow('img'+str(count),resized )
                count += 1
                if iteration%100==0:
                   resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # convert to gray scale
                   cv2.imwrite('found.png', resized )
                   print('clicked:',iteration)
            k = cv2.waitKey(30) & 0xff

            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
#inputMyFace()
def read_image(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def buildModelArchitecture(dim1,dim2):
    def build_base_network(input_shape):
        seq = Sequential()

        nb_filter = [6, 12]
        kernel_size = 3

        # convolutional layer 1
        seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                              border_mode='valid', dim_ordering='th'))
        seq.add(Activation('relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(.25))

        # convolutional layer 2
        seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
        seq.add(Activation('relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
        seq.add(Dropout(.25))

        # flatten
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(50, activation='relu'))
        return seq

    input_dim = (1, dim1, dim2)
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_vecs_a = base_network(input_a)
    feat_vecs_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])


    rms = RMSprop()

    model = Model(input=[input_a, input_b], output=distance)

    model.compile(loss=contrastive_loss, optimizer=rms)
    return model


def trainMyModel():

 #Image.open("orl_faces/s1/1.pgm")




 def get_data(size, total_sample_size):
    # read the image
    image = read_image('orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    # reduce the size
    image = image[::size, ::size]
    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    count = 0
    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    for i in range(40):
        print("generating genuine pairs:",i)
        for j in range(int(total_sample_size / 40)):
            ind1 = 0
            ind2 = 0

            # read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            # read the two images
            img1 = read_image('orl_faces/s' + str(i + 1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('orl_faces/s' + str(i + 1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            # store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            # as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size / 10)):
        print("generating imposite pairs:", i)
        for j in range(10):

            # read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            img1 = read_image('orl_faces/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('orl_faces/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            # as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1

    # now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)
    #np.save("X.npy",X)
    #np.save("Y.npy", Y)
    return X,Y
 X,Y= get_data(size, total_sample_size)
 print(X.shape)
 x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)






 model=buildModelArchitecture( x_train.shape[3],x_train.shape[4])

 print('the layer of the model are:')
 for count,layer in enumerate(model.layers):
     print(count,layer)


 img_1 = x_train[:, 0]
 img_2 = x_train[:, 1]
 epochs = 13
 model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, nb_epoch=epochs)
 model.save_weights('face_recognizer.h5')
 #model.load_weights('face_recognizer.h5')
 #pred = model.predict([x_test[:, 0], x_test[:, 1]])
 #def compute_accuracy(predictions, labels):
 #    return labels[predictions.ravel()]

 #sklearn.metrics.accuracy_score(pred, y_test, normalize=True, sample_weight=None)'
 #for i in range(len(pred)):
 #     print(pred[i],y_test[i])
 #compute_accuracy(pred, y_test)


def predictFaces(pathToImage):
    image = read_image('orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    # reduce the size
    image = image[::size, ::size]
    # get the new size
    dim1,dim2 = image.shape
    model=buildModelArchitecture(dim1, dim2)
    print('the layer of the model are:')
    for count, layer in enumerate(model.layers):
        print(count, layer)

    model.load_weights('face_recognizer.h5')

    def who_is_it(model, pathToImage):
        imageVec = np.zeros(40)
        for i in range(40):
            print("generating imageVector:", i)
            img1 = read_image('orl_faces/s' + str(i + 1) + '/' + str(1) + '.pgm', 'rw+')
            img = read_image(pathToImage)
            img1 = img1[::size, ::size]
            img = img[::size, ::size]
            dim1 = img1.shape[0]
            dim2 = img1.shape[1]
            x_img = np.zeros([1, 2, 1, dim1, dim2])
            x_img[0, 0, 0, :, :] = img1
            x_img[0, 1, 0, :, :] = img

            imageVec[i] = model.predict([x_img[:, 0, :, :, :], x_img[:, 1, :, :, :]])
        return imageVec

    arr = who_is_it(model,pathToImage )
    print('The recognized person is:',np.argmin(arr) + 1)
#trainMyModel()
predictFaces('orl_faces/s25/7.pgm')



