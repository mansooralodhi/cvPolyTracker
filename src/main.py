import cv2
import utils
from object_detector import ObjectDetector
from centroid_tracker import CentroidTracker


def execute(source='recording'):

    # todo: set the below hyper-parameters
    maxDisappeared = 5
    min_object_distance = 40

    object_tracker = CentroidTracker(maxDisappeared)
    object_detector = ObjectDetector(min_object_distance)

    if source == 'recording':
        file = '../demo/exp2.mp4'
        video_stream = cv2.VideoCapture(file)
    else:
        video_stream = cv2.VideoCapture(0)

    while True:
        ret, frame = video_stream.read()

        if not ret:
            break

        # downsize image
        resize_img, _ = utils.resize_img(frame, new_width=520)

        # detect objects
        objects_info = object_detector.detect(resize_img)

        # recognize objects
        objects = object_tracker.update(objects_info.get('centroids'))

        # update object ID
        object_tracker.plot_objectID(objects, resize_img)

        # show the output frame
        cv2.imshow("Frame", resize_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == '__main__':
    execute()
