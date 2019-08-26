import cv2
import glob
import numpy as np

General100 = [img for img in glob.glob("data/train/General-100/*.bmp")]
ninetyOne=[img for img in glob.glob("data/train/91/*.bmp")]
filename=General100+ninetyOne


dim=(100,100)
train_size=3   #change the train size if you want more  data augmentation(more flipping and more random jpeg compressions)

#prepare training data by resizing ,data augmentation and storing in a folder
def generate_training_examples():
 for img in enumerate(filename):
    image= cv2.imread(img[1])
    image=cv2.resize(image, dim)

    for i in range(train_size):
       flipped=cv2.flip(image,np.random.randint(-1,2))       #randomly flip the data along bothe the axis
       cv2.imwrite("training_data/" + str(img[0])+ "_original_" + str(i)  + ".jpg", flipped)
       #apply rrandom compressions from 30 percent to 60 percent quality
       cv2.imwrite("training_data/" + str(img[0]) + "_reduced_" + str(i) + ".jpg", flipped,[cv2.IMWRITE_JPEG_QUALITY, np.random.randint(30,60)])
    print("writing image: "+str(img[0]) )
 return True

#prepare the training matrix X and Y from the data stored,it is called by the train.py
def get_data():
    X = np.zeros((len(filename)*train_size,dim[0], dim[1],3))
    Y = np.zeros((len(filename)*train_size, dim[0], dim[1],3))
    print("size of X is",len(filename)*train_size)
    for img in enumerate(X):
        try:
         for i in range(train_size):
          print("Stroing training_data/" + str(img[0]) + "_reduced_"+str(i)+".jpg to X index:",train_size*img[0]+i)
          print("Stroing training_data/" + str(img[0]) + "_original_" + str(i) + ".jpg to Y index:",train_size * img[0] + i)
          imageX=(cv2.imread("training_data/" + str(img[0]) + "_reduced_"+str(i)+".jpg"))
          imageY=(cv2.imread("training_data/" + str(img[0]) + "_original_" + str(i)+ ".jpg"))
          X[train_size*img[0]+i,:,:,:]=imageX/255
          Y[train_size*img[0]+i, :, :,:] = imageY/255
        except:
            break


    print(Y)
    return X,Y
if __name__=="__main__":
 generate_training_examples()






