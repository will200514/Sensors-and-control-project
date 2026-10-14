"""ArUco detection: camera image -> [(marker id, u, s)], the measurement the EKF already uses.
u = pixel column of the marker centre (gives bearing), s = marker size in pixels (gives range).
Dictionary and detector as in the Week 3 tutorial. Used by MuJoCo now and the real camera later.
"""
import numpy as np
import cv2
from .. import config as C


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX     # sub-pixel corners
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def detect_markers(gray, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for c, j in zip(corners, ids.ravel()):
        if j >= len(C.MARKERS):                  # not one of our markers
            continue
        tl, tr, br, bl = c.reshape(4, 2)
        u = c.reshape(4, 2)[:, 0].mean()
        # Use the vertical edges: they don't shrink when the marker is seen at an angle
        s_vertical = 0.5 * (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br))
        # Off-centre markers are further along the camera axis and look smaller.
        # cos(bearing) converts to the face-on size the EKF model expects (else +15 % range at the image edge).
        bearing = np.arctan((C.CX_PX - u) / C.FX)
        out.append((int(j), float(u), float(s_vertical * np.cos(bearing))))
    return out
