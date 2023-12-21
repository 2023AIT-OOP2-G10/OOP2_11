import cv2

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,100,200)

cv2.imwrite('data/dst/opencv_canny.jpg', edges)