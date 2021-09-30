import cv2 
import matplotlib.pyplot as plt

img= cv2.imread('Photos/park.jpg')
cv2.imshow('Crack',img)
#BGR to Gray
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray_img)
#BGR to HSV

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('HSV',hsv)

#BGR to LAB
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow('LAB',lab)

#BGR to RGB
rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(rgb)
plt.show()


cv2.waitKey(0)