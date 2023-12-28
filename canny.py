import cv2


def canny_process_image(image_path, output_path):
    img = cv2.imread(image_path, 0)
    edges = cv2.Canny(img, 100, 200)

    cv2.imwrite(output_path, edges)
