import cv2


class IdentifyShapes(object):
    def __init__(self):
        pass

    @staticmethod
    def identify(contours):

        shape = 'unknown'
        size = 'unknown'

        perimeter = cv2.arcLength(contours, True)
        # approximation = cv2.approxPolyDP(contours, 0.04 * perimeter, True)
        approximation = cv2.approxPolyDP(contours, 0.02 * perimeter, True)

        cont_len = len(approximation)
        cont_area = cv2.contourArea(approximation)

        if cont_len == 3:
            shape = 'triangle'
            if cont_area <= 3000:
                size = 'small'
            elif 3000 < cont_area <= 9000:
                size = 'medium'
            else:
                size = 'large'

        elif cont_len == 4:
            shape = 'rectangle'
            (_, _, w, h) = cv2.boundingRect(contours)
            aspect_ratio = w / float(h)
            if 0.90 <= aspect_ratio <= 1.10:
                shape = 'square'

            if cont_area <= 18000:
                size = 'small'
            elif 18000 < cont_area < 23000:
                size = 'medium'
            else:
                size = 'large'

        elif cont_len == 5:
            shape = 'pentagon'
            if cont_area <= 6000:
                size = 'small'
            elif 6000 < cont_area < 9000:
                size = 'medium'
            else:
                size = 'large'

        else:
            shape = 'circle'
            if cont_area <= 10000:
                size = 'small'
            elif 10000 < cont_area < 15000:
                size = 'medium'
            else:
                size = 'large'

        return shape, size



