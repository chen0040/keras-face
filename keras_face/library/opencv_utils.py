import numpy as np
import cv2


def detect_face_from_img_path(frontal_face_model_file_path, image_path):
    face_cascade = cv2.CascadeClassifier(frontal_face_model_file_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    frontal_face_model_file_path = '../../demo/opencv-files/haarcascade_frontalface_alt.xml'

    detect_face_from_img_path(
        frontal_face_model_file_path,
        '../../demo/data/images/camera_3.jpg')


if __name__ == '__main__':
    main()
