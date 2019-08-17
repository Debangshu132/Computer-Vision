import cv2
import time
import numpy as np


# download the xml file from here: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')       #classifier to recognize face
cap = cv2.VideoCapture(0)                                                         #capture webcam video
while 1:
    ret, img = cap.read()         #read a frame
    cv2.resize(img, (500,400), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                  #convert to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                           #detect all the faces present on the image
    cv2.imshow('img', img)
    cv2.moveWindow('img',700,00)
    if len(faces)>0:                                                              #only if faces are found save the faces in the video

     #time.sleep(1.5)
     count=0
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
