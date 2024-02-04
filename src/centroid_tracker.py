from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

"""

Source
------
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

Algorithm
---------
Generally Object Tracking can be divided into three phases:
1. Detect the bounding box of object 
2. Assign unique ID to that bounding box
3. Track that unique ID in between frames

The above steps are not that simple to implement and we need further steps:
1. Apply object detector after every 10 frames
2. Apply centroid tracker on each frame
3. Restrict the euclidean distance between object between frames, lets say x cm
4. Register a new object without any unique ID
5. Update the new object location (x,y) on every frame
6. DeRegister the unique ID or object if its not within the euclidean distance within n frames
    It is assumed that the distance between the centroids of same object between frame F(t) and F(t+1) 
    is less then all other distances between objects. 
Below code use the above 6 steps to track an object.


Data Structures
---------------
1. Ordered Dictionary: we use this data structure to store object id as key and its location as value.


Variables
---------
1. nextObjectID: a counter which increments with new objects and whose value is assigned as objectID
2. objects:      an ordered dictionary which maps an objects objectID to its centroid/location
3. disappeared:  an ordered dictionary which maps an objects objectID to the no. of frames 
                 since it has not been found (disappeared). 
4. maxDisappeared: maximum frames allowed for the object to remain disappeared while being marked as 
                    present to user/viewer. It directly related to disappeared dictionary.
                

"""


class CentroidTracker(object):

    def __init__(self, maxDisappeared=5):
        # A counter used to assign unique IDs to each object
        self.nextObjectID = 0
        # no of frames before a registered object is de-registered if not found
        self.maxDisappeared = maxDisappeared
        # Record the objects by mapping their centroids (value) to their  object ID (key)
        self.objects = OrderedDict()
        # Record the number of consecutive frames (value) a particular object ID (key) has not been found
        self.disappeared = OrderedDict()

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, centroids):
        """
        :param centroids: List[centroid1(x,y), centroid2(x,y), ....]
        :return:
        """
        if len(centroids) == 0:
            # it means no object has been detected in this frame, therefore, increment all
            # objects to be disappeared.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive  frames where a given object has
                # been marked as missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(centroids), 2), dtype="int")

        for i, cent in enumerate(centroids):
            inputCentroids[i] = cent

        if len(self.objects) == 0:
            # this is the case if we are calling the update for the first time.
            # or it's the first frame since we have start updating in video
            # it will register any detected centroids
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # otherwise, are currently tracking objects, therefore, we need to try to match the
            # input centroids to existing object centroids

            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids,
            # respectively -- shape(D) = (len(objectCentroids) , len(inputCentroids))
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # todo: understand the below code
            # we find the minimum in each row -> sort the array -> find the arguments of sorted array
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by finding the smallest
            # value in each column and then sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

    @staticmethod
    def plot_objectID(objects, frame):
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)