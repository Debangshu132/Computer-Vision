 
from keras.models import Sequential
import sys
from train_data import build_model,batch_generator
import numpy as np
import cv2
import glob
np.set_printoptions(threshold=sys.maxsize)
import math
import time
import tensorflow as tf
import argparse
scale=2
DOWNGRADED_FOLDER="data/validation/Set5"
ORIGINAL_FOLDER="data/validation/Set5"
parser = argparse.ArgumentParser()
MODEL_NAME="model.h5"
#parser.add_argument("--model_name", help="which model to use for predictions")
#args = parser.parse_args()
#print(args)
DESTINATION_FOLDER="pred/"

def predict(ORIGINAL_FOLDER,DESTINATION_FOLDER):
    model = build_model(2)  # get model architecture
    model.load_weights(MODEL_NAME)  # load previously trained weights
    count = 0
    for image in glob.glob(ORIGINAL_FOLDER+"/*"):
     count+=1
     image = cv2.imread(image)

     original=image
     scale_factor = 1
     #image = cv2.resize(image, (image.shape[1] // scale_factor, image.shape[0] // scale_factor))
     #image = cv2.resize(image, (original.shape[0] , original.shape[1] ))
     image_downgraded = image
     image_downgraded  = cv2.cvtColor(image_downgraded , cv2.COLOR_BGR2YCR_CB)
     image_downgraded_copy = image_downgraded
     image_downgraded = image_downgraded [:, :, 0]
     prediction=break_predict_stitch(model,image_downgraded)


     for img in prediction:
         img[:, :, 0] =img[:,:,0]*255

         image_downgraded_copy[:,:,0]=np.zeros((1,img[:,:,0].shape[0], img[:,:,0].shape[1]))
         img[:, :, 0][img[:, :, 0]<0]=0
         img[:, :, 0][img[:, :, 0] >255] = 255
         image_downgraded_copy[:,:,0] =img[:, :, 0]

         cv2.imwrite(DESTINATION_FOLDER+"/"+str(count) + "_actual.png", original)
         print(image.shape)
         blurred=cv2.blur(image,(3,3),)
         predicted=cv2.cvtColor(image_downgraded_copy, cv2.COLOR_YCrCb2BGR)
         cv2.imwrite(DESTINATION_FOLDER + "/" + str(count) + "_reduced.png", image)
         cv2.imwrite(DESTINATION_FOLDER + "/" + str(count) + "_predicted.jpg",predicted )

         try:
          print(psnr( cv2.cvtColor(original , cv2.COLOR_BGR2YCR_CB)[:, :, 0],image_downgraded_copy[:, :, 0]))
         except:
             continue
def break_predict_stitch(model,image_downgraded):
    X = np.zeros((1, image_downgraded.shape[0], image_downgraded.shape[1], 1))
    X[0, :, :, 0] = 1.0*image_downgraded / 255
    prediction=X
    #prediction = model.predict(X, verbose=1)
    predict_patch_size=20
    full_x=int(image_downgraded.shape[0]/predict_patch_size)
    full_y = int(image_downgraded.shape[1] / predict_patch_size)
    residue_X=image_downgraded.shape[0]%predict_patch_size
    residue_Y = image_downgraded.shape[1] % predict_patch_size
    countX=0
    countY=0
    for countX in range(full_x):
      for countY in range(full_y):
          temp = np.zeros((1, predict_patch_size,predict_patch_size, 1))
          temp[0,:,:,0]=X[0, countX*predict_patch_size:countX*predict_patch_size+predict_patch_size, countY*predict_patch_size:countY*predict_patch_size+predict_patch_size, 0]
          temp = model.predict(temp, verbose=1)
          prediction[0, countX*predict_patch_size:countX*predict_patch_size+predict_patch_size, countY*predict_patch_size:countY*predict_patch_size+predict_patch_size, :]=temp
      temp = np.zeros((1,  residue_X, predict_patch_size, 1))
      temp[0, :, :, 0] = X[0, countX * predict_patch_size:countX * predict_patch_size +  residue_X, countY * predict_patch_size:countY * predict_patch_size + predict_patch_size, 0]
      temp = model.predict(temp, verbose=1)
      prediction[0, countX * predict_patch_size:countX * predict_patch_size +  residue_X, countY * predict_patch_size:countY * predict_patch_size + predict_patch_size, :] = temp
      time.sleep(0.1)
    print(prediction.shape)

    return prediction
def pred(img,model):
    img = cv2.imread(img)
    #img = cv2.resize(img, (200, 200))
    original = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    image_copy = img
    img = img[:, :, 0]
    print( 1.0*img / 255)
    X = np.zeros((1, img.shape[0], img.shape[1], 1))
    X[0, :, :, 0] = 1.0*img / 255
    
    prediction = model.predict(X, verbose=0)
    for img in prediction:
        
        img[:, :, 0] = img[:, :, 0] *255
        img[:, :, 0][img[:, :, 0] < 0] = 0
        img[:, :, 0][img[:, :, 0] > 255] = 255
        image_copy[:, :, 0] = img[:, :, 0]
        predicted = cv2.cvtColor(image_copy, cv2.COLOR_YCrCb2BGR)
        #print(tf.image.psnr(original, predicted, 1, name=None))
        
        vis = np.concatenate((original, predicted), axis=1)
        cv2.imshow("imag",vis)
        cv2.waitKey()








def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
     return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def cam():
    cape = cv2.VideoCapture('c.mp4')
    #cape.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    #cape.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    cape.set(cv2.CAP_PROP_FPS, 5)

    model = build_model()  # get model architecture
    model.load_weights(MODEL_NAME)  # load previously trained weights
    ret, frameinit = cape.read()
    scale_factor = 2
    frameinit = cv2.resize(frameinit, (frameinit.shape[1] // scale_factor, frameinit.shape[0] // scale_factor))

    image_downgraded = cv2.cvtColor(frameinit, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    #image_downgraded = image_downgraded[:, :, 0]

    X = np.zeros((1, image_downgraded.shape[0], image_downgraded.shape[1], 1))
    out = cv2.VideoWriter('c_modified.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frameinit.shape[1], frameinit.shape[0]))
    out_actual = cv2.VideoWriter('c_actual.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frameinit.shape[1], frameinit.shape[0]))
    while (True):
        # Capture frame-by-frame
        time.sleep(0.1)
        ret, frame = cape.read()

        frame = cv2.resize(frame, (frame.shape[1] // scale_factor, frame.shape[0]// scale_factor ))
        frame_ycrcb=cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        X[0, :, :, 0] = frame_ycrcb[:, :, 0]/ 255
        #X[0, :, :, 0] = image_downgraded
        for img in model.predict(X, verbose=1):
              img[:, :, 0][img[:, :, 0] < 0] = 0
              img[:, :, 0][img[:, :, 0] > 255] = 255
              frame_ycrcb[:, :, 0] = img[:, :, 0] * 255
              frame_copy = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
              out.write(frame_copy)
              cv2.imshow('modified', frame_copy)
        out_actual.write(frame)
        cv2.imshow('actual', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            out_actual.release()
            break

    # When everything done, release the capture


    #for i in range(len(img_array)):
    #    out.write(img_array[i])

    cape.release()
    cv2.destroyAllWindows()


def prepare_training_data_from_video(name):
    frame_count=0
    cape = cv2.VideoCapture(name)
    while(True):
     ret, frame = cape.read()
     if frame_count%20000==0:
      cv2.imwrite("youtube/a_"+str(frame_count)+".png",frame)
      print("writing frame number :" , frame_count)
     frame_count+=1






if __name__=="__main__":
 model = build_model(2)  # get model architecture
 model.load_weights(MODEL_NAME)  # load previously trained weights
 #pred("data/validation/Set5/baby.png", model)
 prediction=predict(ORIGINAL_FOLDER,DESTINATION_FOLDER)

 #count=0
 #cam()
 #prepare_training_data_from_video('a.mp4')


