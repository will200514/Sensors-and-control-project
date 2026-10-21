"""Time-varying LQR trajectory tracker acting on the ESTIMATED pose only.
uses the estimated pose from the EKF
create trajectory generation
compare estimate and trajectory and calculates the movement comands
"""
import numpy as np
from .. import config as C
from ..model.diffdrive import error_model, c2d, dlqr, wrap


class LQRTracker:

    # sets up the controller
    # builds the LQR weighting matrices
    # Q determines how much the cvontroller penalise tracking error (forward, sideways, heading)
    # smaller acceptable error = larger penalty
    # R penalises against control corrections
    # du = [forward_speed_correction, turning_speed_correction]
    def __init__(self, cfg):
        self.Q = cfg.lqr_q_scale * np.diag(1.0 / np.array(cfg.e_max) ** 2)    # Bryson's rule
        self.R = cfg.lqr_r_scale * np.diag(1.0 / np.array(cfg.u_max) ** 2)
        self._cache = {}
        self.idx = 0
        self.traj = None

    
    def set_trajectory(self, traj):
        self.traj, self.idx = traj, 0
        self._nearest_idx = 0

    def gain(self, v_r, w_r):
        key = (round(v_r / 0.01), round(w_r / 0.05))
        if key not in self._cache:
            A, B = error_model(max(key[0] * 0.01, 0.01), key[1] * 0.05)
            Ad, Bd = c2d(A, B, C.DT)
            self._cache[key] = dlqr(Ad, Bd, self.Q, self.R)[0]
        return self._cache[key]

    def done(self):
        return self.traj is not None and self.idx >= len(self.traj) - 1

    def command(self, x_est):
        T = self.traj
        lo = self._nearest_idx
        hi = min(lo + 40, len(T))
        d = np.hypot(T[lo:hi, 0] - x_est[0], T[lo:hi, 1] - x_est[1])
        self._nearest_idx = lo + int(np.argmin(d))
        self.idx = min(self._nearest_idx + 8, len(T) - 1)  # no false progress while stationary
        xr, yr, thr, vr, wr = T[self.idx]
        c, s = np.cos(x_est[2]), np.sin(x_est[2])
        dx, dy = xr - x_est[0], yr - x_est[1]
        e = np.array([c * dx + s * dy, -s * dx + c * dy, wrap(thr - x_est[2])])
        # Acquisition may finish facing away from the path: align before driving.
        target_bearing = float(wrap(np.arctan2(dy, dx) - x_est[2]))
        if abs(target_bearing) > 0.6:
            return (0.0, float(np.clip(1.5 * target_bearing, -C.W_MAX, C.W_MAX))), e
        du = -self.gain(vr, wr) @ e
        v = np.clip(vr + du[0], 0.0, C.V_MAX)
        w = np.clip(wr + du[1], -C.W_MAX, C.W_MAX)
        return (v, w), e
