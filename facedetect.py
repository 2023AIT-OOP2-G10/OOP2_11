import cv2
import os

# Haar Cascadeファイルのパス
CASCADE_PATH = "OOP2_11/haarcascade_frontalface/haarcascade_frontalface.xml"


# 顔検出関数





def process_image(file_path, processed_folder):
    processed_path = os.path.join(processed_folder, os.path.basename(file_path))
    detect_faces(file_path, processed_path)
