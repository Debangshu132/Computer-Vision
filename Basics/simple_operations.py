#import all the libraries
import cv2

PATH_TO_FILE="../Images/lena.jpg"
def readImage():
 img=cv2.imread(PATH_TO_FILE) #read the image from file
 cv2.imshow("img",img)                #Show the image
 cv2.waitKey(0)                       #wait until any key is pressed
 cv2.destroyAllWindows()              #destroy all opened windows
#readImage()

#resize an image
def resizeImage():
 img=cv2.imread(PATH_TO_FILE)
 dimensions=(100,100)
 img=cv2.resize(img,dimensions)
 cv2.imshow("img",img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
#resizeImage()

def gray():
    img = cv2.imread(PATH_TO_FILE)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
gray()

def gray_and_blur():
    img = cv2.imread(PATH_TO_FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray image
    img = cv2.GaussianBlur(img)                 #Blur the image
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
gray_and_blur()


