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


def _segment_hits_circle(p0, p1, c, r):
    d = p1 - p0
    t = np.clip(np.dot(c - p0, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p0 + t * d - c) < r


def observe_markers(x, obstacles, rng, fov_deg=62.2):
    """Simulated detector. Returns a list of (marker_index, u_px, s_px) using TRUE pose x.
    A marker is detected if it is in range, inside the field of view, seen from the front
    (incidence < 60 deg) and not occluded by an obstacle."""
    out = []
    cam = cam_position(x)
    half = np.deg2rad(fov_deg) / 2
    for j, (mx, my, nang) in enumerate(C.MARKERS):
        mk = np.array([mx, my])
        d = mk - cam
        rho = np.linalg.norm(d)
        if not (C.CAM_MIN_RANGE < rho < C.CAM_MAX_RANGE):
            continue
        beta = wrap(np.arctan2(d[1], d[0]) - x[2])
        if abs(beta) > half:
            continue
        incidence = abs(wrap(np.arctan2(-d[1], -d[0]) - nang))   # marker normal vs ray to camera
        if incidence > C.MARKER_MAX_INCIDENCE:
            continue
        if any(_segment_hits_circle(cam, mk, np.array(o[:2]), o[2]) for o in obstacles):
            continue
        beta_n = beta + rng.normal(0, C.SIGMA_BEARING_FLOOR)
        u = C.CX_PX - C.FX * np.tan(beta_n) + rng.normal(0, C.SIGMA_PX)
        s = C.FX * C.MARKER_L / rho + rng.normal(0, C.SIGMA_SIZE_PX)
        if s > 2.0:
            out.append((j, u, s))
    return out
