import cv2
import numpy as np


def remove_noise(img, kernel=None):
    if kernel is None:
        kernel = np.ones((3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def resize_img(img, new_width=None):
    height, width = img.shape[0], img.shape[1]
    aspect_ratio = new_width / width
    new_height = int(height * aspect_ratio)
    new_img = cv2.resize(img, (new_width, new_height))
    return new_img, aspect_ratio

