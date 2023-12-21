import cv2
import os

# Haar Cascadeファイルのパス
CASCADE_PATH = "OOP2_11/haarcascade_frontalface/haarcascade_frontalface.xml"

# 顔検出関数
def detect_faces(image_path, output_path):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imwrite(output_path, image)

# 既存のprocess_image関数を変更
def process_image(file_path):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
    detect_faces(file_path, processed_path)
