import cv2
import numpy as np
img = cv2.imread('Photos/group 2.jpg')

def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv2.warpAffine(img,transMat,dimensions)


# -X --> Left
# -y-->Up
translated = translate(img,100,100)

# cv2.imshow("Translated",translated)

# Rotaation 
def rotate(img,angle,rotPoint= None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint= (width//2,height//2)
    rotMat = cv2.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)
    return cv2.warpAffine(img,rotMat,dimensions)
rotated = rotate(img,45)
# cv2.imshow('Rotated',rotated)


# Resize 
resized = cv2.resize(img, (500,500), interpolation  = cv2.INTER_CUBIC)
# cv2.imshow('Resized',resized)

cv2.imshow('normal',img)
# Flipped
flip = cv2.flip(img,1)
cv2.imshow('flipped',flip)

cv2.waitKey(0)
cv2.destroyAllWindows()