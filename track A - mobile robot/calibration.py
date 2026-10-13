"""Camera intrinsic calibration (fx, cx) from views of a marker at known poses.

Offline step: the robot is placed at known poses in front of a marker (like a calibration board),
so geometry is known by construction. Each detection gives two linear relations:
    u      = cx - fx * tan(beta)          (bearing equation)
    s      = (fx L) * (1 / rho)           (size equation)
Both are linear in the unknowns, so ordinary least squares gives (fx, cx) and (fx L) with
covariance, and we compare fx from the two equations as a consistency check.
"""
import numpy as np
from .. import config as C
from .camera import cam_position
from ..model.diffdrive import wrap


def calibrate(n_views=40, seed=0, fx_true=C.FX, cx_true=C.CX_PX):
    rng = np.random.default_rng(seed)
    mk = np.array([2.5, 0.02])
    rows_u, rhs_u, rows_s, rhs_s = [], [], [], []
    for _ in range(n_views):
        x = np.array([rng.uniform(1.5, 3.5), rng.uniform(0.6, 1.8), rng.uniform(np.deg2rad(-115), np.deg2rad(-65))])
        d = mk - cam_position(x)
        rho = np.linalg.norm(d)
        beta = wrap(np.arctan2(d[1], d[0]) - x[2])
        if abs(beta) > np.deg2rad(28):
            continue
        u = cx_true - fx_true * np.tan(beta) + rng.normal(0, C.SIGMA_PX)
        s = fx_true * C.MARKER_L / rho + rng.normal(0, C.SIGMA_SIZE_PX)
        rows_u.append([-np.tan(beta), 1.0]); rhs_u.append(u)          # u = fx*(-tan b) + cx
        rows_s.append([1.0 / rho]); rhs_s.append(s)                   # s = (fx L) / rho
    Au, bu = np.array(rows_u), np.array(rhs_u)
    (fx_u, cx), *_ = np.linalg.lstsq(Au, bu, rcond=None)
    cov = np.linalg.inv(Au.T @ Au) * C.SIGMA_PX ** 2
    As, bs = np.array(rows_s), np.array(rhs_s)
    (fxL,), *_ = np.linalg.lstsq(As, bs, rcond=None)
    return dict(fx_bearing=fx_u, cx=cx, fx_size=fxL / C.MARKER_L, n=len(bu),
                fx_std=float(np.sqrt(cov[0, 0])), cx_std=float(np.sqrt(cov[1, 1])))
