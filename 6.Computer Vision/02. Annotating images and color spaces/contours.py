import cv2 
import numpy as np

img  = cv2.imread('Photos/cats.jpg')

cv2.imshow("Cats",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray',gray)

blur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
cv2.imshow('Blurred',blur)
# # Question 
# # What do we need destroyAllWindows if we can close it by just waitKey

canny = cv2.Canny(img,125,175)


cv2.imshow('Canny',canny)

# def sort_contours(cnts, method="left-to-right"):
#     # initialize the reverse flag and sort index
#     reverse = False
#     i = 0
#     # handle if we need to sort in reverse
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True

#     # handle if we are sorting against the y-coordinate rather than
#     # the x-coordinate of the bounding box
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1

#     # construct the list of bounding boxes and sort them from top to
#     # bottom
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
#                                         key=lambda b: b[1][i], reverse=reverse))
#     # return the list of sorted contours and bounding boxes
#     return (cnts, boundingBoxes)


ret,thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)

contours, hierarchies = cv2.findContours(canny,
                        cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

print(f"{len(contours)} contours found")
cv2.waitKey(0)

