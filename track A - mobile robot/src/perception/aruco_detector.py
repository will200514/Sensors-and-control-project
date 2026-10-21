"""Detect the single docking ID and preserve its four ordered image corners.

Corners are top-left, top-right, bottom-right, bottom-left in marker coordinates.
No simulator marker coordinates are used by this detector.
"""
import cv2
import numpy as np
from .. import config as C


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(gray, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(j), np.asarray(c, dtype=float).reshape(4, 2))
            for c, j in zip(corners, ids.ravel()) if int(j) == C.DOCK_MARKER_ID]
