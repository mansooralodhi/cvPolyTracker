import cv2
import utils
from _object_detector import ObjectDetector
from _extract_roi import ExtractROI

"""
Source: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
"""


def unit_test():
    cap = cv2.VideoCapture('../solar.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def extract_roi(file):
    cap = cv2.VideoCapture(file)
    roiExtractor = ExtractROI()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    pass


def detect_objects(file):
    cap = cv2.VideoCapture(file)
    ratio = None
    obj_detector = ObjectDetector(ratio)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        ############################  Custom Code ##############################
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = utils.remove_noise(gray_img)
        bin_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)[1]

        resized_img, ratio = utils.resize_img(bin_img, (300, 300))
        obj_detector.resize_ratio = ratio
        frame = obj_detector.detect(resized_img, frame, show=False)

        ############################    END     ###############################

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # detect_objects('videos/VID_20221114_172730.mp4')
    detect_objects('videos/Snapsave.app_130221091_211908663783562_3811067093687436181_n.mp4')
