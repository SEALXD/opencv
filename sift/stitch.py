import tt
import cv2 as cv
import imutils

img1 = cv.imread('right.jpg')
img2 = cv.imread('left.jpg')
img1 = imutils.resize(img1, width=400) #缩放为同尺寸
img2 = imutils.resize(img2, width=400)

ratio = 0.8
reprojThresh = 4.0

(result, vis) = tt.stitch(img1,img2, ratio,reprojThresh,showMatches=True)


cv.imshow('image A', img1)
cv.imshow('image B', img2)
cv.imshow('keyPoint Matches', vis)
cv.imshow('Result', result)

cv.waitKey(0)
cv.destroyAllWindows()