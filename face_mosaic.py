# face_mosaic.py

import cv2
import os
from PIL import Image

def detect_and_mosaic_faces(file_path, output_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (40, 40))
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_AREA)
        image[y:y + h, x:x + w] = face

    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    # テスト用のコード
    input_path = 'path/to/input/image.jpg'
    output_path = 'path/to/output/image_mosaic.jpg'

    detect_and_mosaic_faces(input_path, output_path)