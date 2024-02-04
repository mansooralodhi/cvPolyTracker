import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from binarize import Binarize


class Segmentation(Binarize):
    def __init__(self):
        super().__init__()
        self.binarize = Binarize()

    def segment(self, img):
        # gray to binary image
        # ret1, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        thresh = self.binarize.inRangeThresholding(img)

        # remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.erode(opening, kernel, iterations=2)
        # # markers = np.uint8()
        # plt.imshow(sure_bg)
        # plt.show()
        # key = cv2.waitKey(1) & 0xFF

        # distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

        ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

        # unknown ambiguous region is nothing but background - foreground
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # connected component labelling.
        ret3, markers = cv2.connectedComponents(sure_fg)

        # add 10 to all labels so that background is not 0, but 10
        markers = markers + 10
        markers[unknown == 255] = 0

        # watershed filling.
        markers = cv2.watershed(rgb_img, markers)


        markers[markers == -1] = 0
        markers[markers == 10] = 0
        markers[markers != 0] = 255

        # color boundaries in yellow. OpenCv assigns boundaries to -1 after watershed.
        # rgb_img[markers == -1] = [255, 255, 255]
        # img2 = color.label2rgb(markers, bg_label=0)

        markers = np.uint8(markers)
        #
        # markers = np.uint8(markers)
        # cv2.imshow("Frame", markers)
        # key = cv2.waitKey(1) & 0xFF


        return markers
