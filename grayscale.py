import cv2
#import numpy as np

im = cv2.imread('data/src/lena.jpg')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite('data/dst/opencv_gray_cvtcolr.jpg', im_gray)