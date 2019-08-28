from keras.models import Sequential
from train_data import build_model,batch_generator
import numpy as np
import cv2
import glob
import math
import tensorflow as tf

DOWNGRADED_FOLDER="data/validation/set14-downgraded"
ORIGINAL_FOLDER="data/validation/Set14"

MODEL_NAME="model.h5"

DESTINATION_FOLDER="predictions/"

def predict(DOWNGRADED_FOLDER,ORIGINAL_FOLDER,DESTINATION_FOLDER):
    model = build_model()  # get model architecture
    model.load_weights(MODEL_NAME)  # load previously trained weights
    count = 0
    for image_downgraded,image in zip(glob.glob(DOWNGRADED_FOLDER+"/*.png"),glob.glob(ORIGINAL_FOLDER+"/*.png")):
     count+=1
     image = cv2.imread(image)
     image_downgraded = cv2.imread(image_downgraded )
     image_downgraded_copy = image_downgraded
     X=np.zeros((1,image_downgraded .shape[0],image_downgraded .shape[1],3))
     X[0,:,:,:]=image_downgraded
     prediction = model.predict(X, verbose=1)
     for img in prediction:



         cv2.imwrite(DESTINATION_FOLDER+"/"+str(count)+"_predicted.png",img )
         cv2.imwrite(DESTINATION_FOLDER+"/"+str(count) + "_jpeg_compressed.png", image_downgraded_copy)
         cv2.imwrite(DESTINATION_FOLDER+"/"+str(count) + "_actual.png", image)




if __name__=="__main__":

 prediction=predict(DOWNGRADED_FOLDER,ORIGINAL_FOLDER,DESTINATION_FOLDER)
 count=0


