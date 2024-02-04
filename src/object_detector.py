import cv2
import utils
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from identify_shapes import IdentifyShapes
"""
Source: https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
"""


class ObjectDetector(IdentifyShapes):

    def __init__(self, resize_ratio=None,
                 lower_color_bond = np.array([95, 50, 0]),
                 upper_color_bond= np.array([115, 255, 255]),
                 min_distance = 40):

        super().__init__()
        self.resize_ratio = resize_ratio
        self.lower_bond = lower_color_bond
        self.upper_bond = upper_color_bond
        self.min_distance = min_distance

    def preprocess(self, bgr_img):
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        thresh = 255-cv2.inRange(hsv_img, self.lower_bond, self.upper_bond)
        mask = utils.remove_noise(thresh)
        return mask

    def detect(self, actual_img, resize_ratio=(1,1), show=False):
        bin_img = self.preprocess(actual_img.copy())

        D = ndimage.distance_transform_edt(bin_img)
        localMax = peak_local_max(D, indices=False, min_distance=self.min_distance, labels=bin_img)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=bin_img)

        centroids = list()
        filtered_contours = list()

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(bin_img.shape, dtype="uint8")
            mask[labels == label] = 255

            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont = contours[0][0]

            if cv2.contourArea(cont) < 300:
                continue

            moment = cv2.moments(cont)
            cx = int((moment["m10"] / moment["m00"]) * resize_ratio[0])
            cy = int((moment["m01"] / moment["m00"]) * resize_ratio[1])

            shape, size = self.identify(cont)

            cont = cont.astype('float')
            cont[:, :, 0] *= resize_ratio[1]
            cont[:, :, 1] *= resize_ratio[0]
            cont = cont.astype('int')

            centroids.append((cx, cy))
            filtered_contours.append(cont)

            # cv2.drawContours(actual_img, [cont], -1, (0, 255, 0), 2)
            cv2.putText(actual_img, shape, (cx+10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(actual_img, size, (cx+10, cy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if show:
                cv2.imshow('Image', actual_img)
                cv2.waitKey(0)

        return dict(contour_img=actual_img, centroids=centroids, contours=filtered_contours)


if __name__ == '__main__':
    img = cv2.imread('backgrounds/IMG_20221122_010810.jpg')
    obj_detector = ObjectDetector()
    obj_detector.detect(img, show=True)
