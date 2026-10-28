"""Single upright square marker -> planar robot pose in the dock frame.

D: origin below marker, x out of wall, y along wall, z up.
B: robot x forward, y left, z up. C: OpenCV x right, y down, z forward.
M: printed marker x right, y up, z out of its front face.
T_A_B maps coordinates from B into A. No simulation coordinates enter this file.
Assumes a level robot, an upright marker, and the calibrated rigid camera mount.

converts the markers 4 corners into the robot pose expressed relative to the docking marker, 
with the uncertainty of that measurement
conncection between aruco etector and EKF
"""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares
from .. import config as C
from ..model.diffdrive import wrap




# packages the results, i.e. acts as a data container rather than a calculation fuction
@dataclass
class PoseObservation:
    pose: np.ndarray
    covariance: np.ndarray
    reprojection_px: float

# creates the poition and orientation transform for frames
def transform(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T

# describes the camera mounting
# converts camer coordinates into robot body coordinates
def body_camera_transform():
    return transform(np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]),
                     [C.CAM_OFFSET, 0., C.CAM_HEIGHT])


# describes the marker in the docking frame
# converts the aruco markers coordinates into the docking frame coordinates
def dock_marker_transform():
    return transform(np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
                     [0., 0., C.MARKER_HEIGHT])



# defines the physical squere
# provides known physical coordinates of the markers corners
def marker_corners():
    h = C.MARKER_L / 2
    # Required ordering for SOLVEPNP_IPPE_SQUARE.
    return np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])



# descibes ther cameras internal geometry
# includes focal length in pixels, horizontal principle point, verticle principle point
def camera_matrix(fx=C.FX):
    return np.array([[fx, 0., C.CX_PX], [0., fx, C.CY_PX], [0., 0., 1.]])



# predicts what the marker should look like
# i.e if the robot was at this pose relative to the marker, where should the corners appear in the image
def project_corners(pose, fx=C.FX):
    """Forward model: predicted four pixel corners for a robot pose in D."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    T_D_B = transform(np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]), [x, y, 0.])  # camera pose in docking frame
    T_C_M = np.linalg.inv(T_D_B @ body_camera_transform()) @ dock_marker_transform()  # calculates the marker pose relative to the camera
    rvec = cv2.Rodrigues(T_C_M[:3, :3])[0]
    return cv2.projectPoints(marker_corners(), rvec, T_C_M[:3, 3], camera_matrix(fx),  # projects the physical corners into pixels
                             np.array(C.CAM_DISTORTION))[0].reshape(4, 2)


# calculates the measured robot pose
# 1. checks the detected corners
# 2. generates possible marker poses
# 3. convert them into robot poses
# 4. refine using floor constraint
# 5. select the best fit
# 6. calculates the measurement uncertainty
def estimate_dock_pose(corners, cfg):
    """PnP initial candidates, then a level-floor reprojection fit using all corners.

    Reject tiny, behind-marker or poor-reprojection observations. The 8x3 corner
    Jacobian propagates pixel noise into a full 3x3 pose covariance. Small fixed
    floors represent calibration/model uncertainty; r_scale scales the covariance.
    Nearly indistinguishable planar solutions additionally inflate the covariance.
    """
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(pixels).all():
        return None
    edges = np.linalg.norm(pixels - np.roll(pixels, -1, axis=0), axis=1)
    if edges.min() < 6.0:
        return None
    try:
        result = cv2.solvePnPGeneric(marker_corners(), pixels, camera_matrix(cfg.fx_filter),     # estimates the markers otation and translation elative to the camera
                                    np.array(C.CAM_DISTORTION), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    candidates = list(zip(result[1], result[2])) if result[0] else []
    # Iterative planar PnP also covers an exactly fronto-parallel IPPE singularity.
    ok, rv, tv = cv2.solvePnP(marker_corners(), pixels, camera_matrix(cfg.fx_filter),
                             np.array(C.CAM_DISTORTION), flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        candidates.append((rv, tv))
    fits = []
    for rvec, tvec in candidates:
        if tvec[2, 0] <= 0:
            continue
        T_C_M = transform(cv2.Rodrigues(rvec)[0], tvec.ravel())
        T_D_B = dock_marker_transform() @ np.linalg.inv(T_C_M) @ np.linalg.inv(body_camera_transform())
        if T_D_B[0, 3] <= 0.08 or T_D_B[0, 3] >= 20.:
            continue
        initial = [max(T_D_B[0, 3], 0.10), np.clip(T_D_B[1, 3], -15., 15.),
                   np.arctan2(T_D_B[1, 0], T_D_B[0, 0])]
        fit = least_squares(lambda p: (project_corners(p, cfg.fx_filter) - pixels).ravel(),
                            initial, bounds=([0.08, -20., -np.inf], [20., 20., np.inf]),
                            max_nfev=60)
        pose = fit.x.copy()
        pose[2] = wrap(pose[2])
        # The marker must be in front of the onboard camera, not just x_D > 0.
        forward = np.dot(-pose[:2], [np.cos(pose[2]), np.sin(pose[2])]) - C.CAM_OFFSET
        if fit.success and forward > 0.05:
            fits.append((np.sqrt(np.mean(fit.fun ** 2)), pose, fit.jac))
    if not fits:
        return None
    fits.sort(key=lambda item: item[0])
    rms, pose, J = fits[0]
    if rms > cfg.max_reprojection_px:
        return None
    info = J.T @ J
    values, vectors = np.linalg.eigh(info)
    covariance = cfg.corner_sigma_px ** 2 * ((vectors / np.maximum(values, 1e-8)) @ vectors.T)
    for other_rms, other_pose, _ in fits[1:]:
        if other_rms < rms + 0.25:
            delta = other_pose - pose
            delta[2] = wrap(delta[2])
            covariance += 0.25 * np.outer(delta, delta)
    covariance += np.diag([cfg.camera_pos_floor ** 2, cfg.camera_pos_floor ** 2,
                           np.deg2rad(cfg.camera_head_floor_deg) ** 2])
    return PoseObservation(pose, cfg.r_scale * covariance, float(rms))
