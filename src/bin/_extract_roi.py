import cv2
import numpy as np
import utils



def extract_from_img():
    bgr_img = cv2.imread('backgrounds/IMG_20221122_010810.jpg')

    bgr_img = cv2.resize(bgr_img, (540, 320))

    bg_extractor = ExtractROI(find_contour=False, draw_contours=True)
    img = bg_extractor.extract_roi(bgr_img)

    cv2.imshow('frame ', img)
    cv2.waitKey(0)


def extract_from_imgDir():

    import os
    base = 'backgrounds'
    for file in os.listdir(base):
        bgr_img = cv2.imread(base + '/' + file)

        bg_extractor = ExtractROI(find_contour=True, draw_contours=True)
        img = bg_extractor.extract_roi(bgr_img)

        img = cv2.resize(img, (540, 540))
        cv2.imshow('frame ', img)
        cv2.waitKey(0)


def extract_from_video():
    cap = cv2.VideoCapture('videos/VID_20221114_172730.mp4')
    bg_extractor = ExtractROI()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, (540, 540))

        img = bg_extractor.extract_roi(frame)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pass

class ExtractROI(object):
    def __init__(self, draw_contours=True, find_contour=False):
        """
        :param img: (height, width, channels),  channels -> BGR
        :param lower_color_bond: (h_min, s_min, v_min)
        :param upper_color_bond: (h_max, s_max, v_max)
        """
        self.find_contour = find_contour
        self.draw_contours = draw_contours
        self.kernel = np.ones((3, 3), np.uint8)
        self.lower_color_bond = np.array([8, 240, 155])
        self.upper_color_bond = np.array([20, 250, 215])

    def extract_roi(self, img):
        mask = self.segmentation(img)
        if self.find_contour:
            contours = self.get_bg_contours(mask.astype('uint8'))
            cv2.drawContours(img, [contours], -1, (0, 0, 255), 8)
        else:
            return self.separate_foreground(img, mask)
        return img

    @staticmethod
    def separate_foreground(img, mask):
        mask[mask != 0] = 1
        mask = mask.astype('uint8')
        mask = np.stack([mask, mask, mask], axis=2)
        return np.multiply(mask, img)

    @staticmethod
    def get_bg_contours(mask):
        all_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_boundary, lake_cnt_index = 0, 0
        for obj_no in range(len(all_contours)):
            if all_contours[obj_no].shape[0] > max_boundary:
                lake_cnt_index = obj_no
                max_boundary = all_contours[obj_no].shape[0]
        bg_contour = all_contours[lake_cnt_index]

        # todo: plot rectangle in controlled environment
        # (x, y, w, h) = cv2.boundingRect(bg_contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return bg_contour

    def segmentation(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, self.lower_color_bond, self.upper_color_bond) / 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask


if __name__ == '__main__':
    # extract_from_imgDir()
    # extract_from_video()
    extract_from_img()
