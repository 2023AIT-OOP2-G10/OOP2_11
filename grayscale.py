import cv2


# import numpy as np

def gray_process_image(image_path, output_path):
    im = cv2.imread(image_path)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_path, im_gray)
