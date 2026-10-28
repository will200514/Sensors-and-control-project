"""Pinhole camera + ArUco marker measurement model (range and bearing).

Frames: world W, robot B (x forward, origin at wheel-axle centre), camera C (mounted CAM_OFFSET
ahead of B on the x axis, looking along +x_B). A detected marker gives:
    u  : pixel column of the marker centre      -> bearing  beta = atan((cx - u) / fx)
    s  : apparent marker side length in pixels  -> range    rho  = fx * L / s
Both are expressed relative to the CAMERA, so the EKF measurement model includes the offset.
"""
import numpy as np
from .. import config as C
from ..model.diffdrive import wrap


def cam_position(x, offset=C.CAM_OFFSET):
    return np.array([x[0] + offset * np.cos(x[2]), x[1] + offset * np.sin(x[2])])


def h_cam(x, mk_xy, offset=C.CAM_OFFSET):
    """Expected [range, bearing] of marker mk_xy from robot pose x."""
    dx = mk_xy[0] - x[0] - offset * np.cos(x[2])
    dy = mk_xy[1] - x[1] - offset * np.sin(x[2])
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def H_cam(x, mk_xy, offset=C.CAM_OFFSET):
    dx = mk_xy[0] - x[0] - offset * np.cos(x[2])
    dy = mk_xy[1] - x[1] - offset * np.sin(x[2])
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    ddx_dth = offset * np.sin(x[2])
    ddy_dth = -offset * np.cos(x[2])
    return np.array([[-dx / r, -dy / r, (dx * ddx_dth + dy * ddy_dth) / r],
                     [dy / q, -dx / q, (dx * ddy_dth - dy * ddx_dth) / q - 1.0]])


def pixels_to_rb(u, s, fx=C.FX, cx=C.CX_PX, L=C.MARKER_L):
    return fx * L / s, np.arctan((cx - u) / fx)


def R_cam(rho, fx=C.FX, L=C.MARKER_L, sigma_px=C.SIGMA_PX, sigma_s=C.SIGMA_SIZE_PX,
          sigma_b_floor=C.SIGMA_BEARING_FLOOR):
    """Measurement covariance derived from pixel noise by first-order propagation:
        rho = fx L / s  ->  sigma_rho = rho^2 sigma_s / (fx L)      (grows with range squared)
        beta = atan((cx-u)/fx) -> sigma_beta^2 = (sigma_u / fx)^2 + floor^2
    """
    return np.diag([(rho ** 2 * sigma_s / (fx * L)) ** 2,
                    (sigma_px / fx) ** 2 + sigma_b_floor ** 2])