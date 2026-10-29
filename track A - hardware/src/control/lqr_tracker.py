"""Time-varying LQR trajectory tracker acting on the ESTIMATED pose only.
uses the estimated pose from the EKF
create trajectory generation
compare estimate and trajectory and calculates the movement comands
"""
import numpy as np
from .. import config as C
from ..model.diffdrive import error_model, c2d, dlqr, wrap


class LQRTracker:

    # sets up the controllers prioities
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
        # [SMOOTH NAV] turn-in-place hysteresis and lookahead settings
        self.cfg = cfg
        self.turning_in_place = False
        self.cross_track = 0.0


    # loads a new trajectory
    def set_trajectory(self, traj):
        self.traj, self.idx = traj, 0
        self._nearest_idx = 0
        # [SMOOTH NAV] arc length along the path, for a distance-based lookahead
        steps = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))
        self._arc = np.hstack([[0.0], np.cumsum(steps)])
        self.cross_track = 0.0


    # calculates or retreives feedback gain
    # returns the LQR feedback gain K. 
    # 
    def gain(self, v_r, w_r):
        key = (round(v_r / 0.01), round(w_r / 0.05))
        if key not in self._cache:
            A, B = error_model(max(key[0] * 0.01, 0.01), key[1] * 0.05)
            Ad, Bd = c2d(A, B, C.DT)
            self._cache[key] = dlqr(Ad, Bd, self.Q, self.R)[0]
        return self._cache[key]


    # checks wether the active target is the last row
    def done(self):
        return self.traj is not None and self.idx >= len(self.traj) - 1



    # calculates the movement command
    # 1. searches for a nearby future trajectory point
    # 2. choses a target
    # 3. reads the rteferecne, These supply the desired position, heading and baseline speeds at the target.
    # 4. calculates errors in the robots frame
    # 5. oreitns the robot correctly
    # 6. apply LQR feedback and speed limits, 
    def command(self, x_est):
        """Calculate forward and turning commands from the estimated pose."""
        trajectory = self.traj
        x, y, heading = x_est

        cfg = self.cfg
        lookahead = getattr(cfg, "lookahead", None)

        # Find the nearest point within a forward search window.
        # [SMOOTH NAV] Samples are only a few mm apart at hardware speed, so the
        # original 40-sample window is ~12 cm. Search ~1 m when lookahead is on.
        start = self._nearest_idx
        stop = min(start + (400 if lookahead else 40), len(trajectory))

        distances = np.hypot(
            trajectory[start:stop, 0] - x,
            trajectory[start:stop, 1] - y,
        )

        self._nearest_idx = start + int(np.argmin(distances))
        self.cross_track = float(distances.min())

        # Aim slightly ahead of the nearest point.
        self.idx = min(
            self._nearest_idx + 8,
            len(trajectory) - 1,
        )

        xr, yr, heading_ref, v_ref, w_ref = trajectory[self.idx]

        # Express tracking error in the robot frame.
        dx, dy = xr - x, yr - y
        c, s = np.cos(heading), np.sin(heading)

        error = np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
            wrap(heading_ref - heading),
        ])

        # Rotate towards the target before driving if poorly aligned.
        # [SMOOTH NAV] With lookahead set, the bearing is measured to a point
        # `lookahead` metres along the path, not 8 samples (~2 cm on hardware)
        # ahead, which swung wildly with a few cm of position error. Separate
        # enter/exit thresholds stop it flicking between "spin" and "drive".
        if lookahead:
            arc = self._arc
            target = int(np.searchsorted(arc, arc[self._nearest_idx] + lookahead))
            target = min(target, len(trajectory) - 1)
            gx, gy = trajectory[target, 0] - x, trajectory[target, 1] - y
        else:
            gx, gy = dx, dy

        bearing_error = float(wrap(np.arctan2(gy, gx) - heading))

        threshold = (cfg.turn_in_place_exit if self.turning_in_place
                     else cfg.turn_in_place_enter)

        # Bearing to a point a few cm away is meaningless; leave it to the LQR.
        if np.hypot(gx, gy) > 0.08 and abs(bearing_error) > threshold:
            self.turning_in_place = True
            turn = cfg.turn_in_place_gain * bearing_error
            if abs(turn) < cfg.turn_in_place_min_w:
                turn = np.sign(turn) * cfg.turn_in_place_min_w
            turn = float(np.clip(turn, -C.W_MAX, C.W_MAX))
            return (0.0, turn), error

        self.turning_in_place = False

        # Add LQR corrections to the reference speeds.
        correction = -self.gain(v_ref, w_ref) @ error

        forward = np.clip(
            v_ref + correction[0],
            0.0,
            C.V_MAX,
        )
        turn = np.clip(
            w_ref + correction[1],
            -C.W_MAX,
            C.W_MAX,
        )

        return (forward, turn), error
