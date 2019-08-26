from keras.models import Sequential
from train_data import build_model
import numpy as np
import cv2
import glob
import math

#prepare data to evaluate
set5_downgraded = [img for img in glob.glob("data/validation/set5-downgraded/*.png")]
set5= [img for img in glob.glob("data/validation/Set5/*.png")]
set14_downgraded = [img for img in glob.glob("data/validation/set14-downgraded/*.png")]
set14 = [img for img in glob.glob("data/validation/Set14/*.png")]
Xnames=set5_downgraded+set14_downgraded
Ynames=set5+set14



model = build_model((100,100)) #get model architecture
model.load_weights("model.h5") #load previously trained weights
dim=(100,100)
def generate_data_for_evaluation():  #transform the given data into suitable format.We are just the given images to fit the 100 by 100 input of model as of now just to ee the effects)
    for img in enumerate(Xnames):
        image = cv2.imread(Ynames[img[0]])
        image_reduced= cv2.imread(Xnames[img[0]])
        image = image[0:100,0:100]
        image_reduced = image_reduced[0:100,0:100]
        cv2.imwrite("test_data/" + str(img[0]) + ".png", image)
        cv2.imwrite("test_data/" + str(img[0]) + "_reduced.png", image_reduced)
        print("writing image: " + str(img[0]))

#get the data to predict
def get_data_for_evaluation():
        X = np.zeros((len(Xnames), dim[0], dim[1], 3))
        Y = np.zeros((len(Xnames), dim[0], dim[1], 3))
        for img in enumerate(X):
            X[img[0], :, :, :] = cv2.imread("test_data/" + str(img[0]) + "_reduced.png") /255

            Y[img[0], :, :, :] = cv2.imread("test_data/" + str(img[0]) + ".png") / 255
        return X, Y
#actual prediction happens here
def predict(X):
    prediction = model.predict(X, verbose=1)
    prediction = prediction * 255

    psnrarr=[]

    for img in enumerate(prediction):
        cv2.imwrite("test_data/" + str(img[0]) + "_predicted.png", img[1])
        jpeg=cv2.imread("test_data/" + str(img[0]) + "_reduced.png")
        ref=cv2.imread("test_data/" + str(img[0]) + ".png")



        #print('the PSNR is: ', psnr(img[1],ref))
        psnrarr.append(psnr(img[1],ref))


    print("the average psnr is: ",(sum(psnrarr)/len(psnrarr)))

    return prediction
def psnr(target, ref):          #custom definition of psnr
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')


    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)
if __name__=="__main__":
 generate_data_for_evaluation()
 X,Y=get_data_for_evaluation()
 prediction=predict(X)

