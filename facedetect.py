import cv2
import os

# Haar Cascadeファイルのパス
CASCADE_PATH = "haarcascade_frontalface/haarcascade_frontalface_default.xml"


# 顔検出関数
def detect_faces(image_path, output_path):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(output_path, image)


def process_image(file_path, processed_folder):
    processed_path = os.path.join(processed_folder, os.path.basename(file_path))
    detect_faces(file_path, processed_path)
