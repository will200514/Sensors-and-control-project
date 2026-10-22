"""Detect the single docking ID and preserve its four ordered image corners.

Corners are top-left, top-right, bottom-right, bottom-left in marker coordinates.
No simulator marker coordinates are used by this detector.

does not use the markers position in the simulated room
"""
import cv2
import numpy as np
from .. import config as C

# creates and configures the aruco detector
# 
def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX        # subpixel corner refinement
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # uses the aruco dictionary to ccogniose the marker patterns
    return cv2.aruco.ArucoDetector(dictionary, params)


# finds the docking marker in the camera frame
# 
def detect_markers(gray, detector):  
    corners, ids, _ = detector.detectMarkers(gray)  # grey scle camera image
    if ids is None:
        return []     # if no marker detected, robot keeps either searching, navigates with odometry or reacquire the marker
    return [(int(j), np.asarray(c, dtype=float).reshape(4, 2))
            for c, j in zip(corners, ids.ravel()) if int(j) == C.DOCK_MARKER_ID]
