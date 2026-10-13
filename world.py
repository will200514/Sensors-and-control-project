"""Stand-in simulator for the TurtleBot3 room (pure NumPy).

The pipeline only touches this class through: step(v_cmd, w_cmd) -> odometry, camera(), lidar(),
and truth() which is used ONLY for evaluation/logging. A MuJoCo backend would implement the same
four methods (see src/sim/mujoco_backend.py).
"""
import numpy as np
from .. import config as C
from ..model.diffdrive import wheels_to_body, body_to_wheels, f_motion, wrap
from ..perception.camera import observe_markers
from ..perception.lidar import ray_cast


def sample_scenario(rng, cfg):
    start = np.array([rng.uniform(0.4, 1.2), rng.uniform(3.3, 4.6), rng.uniform(-np.pi, np.pi)])
    lead_end = C.DOCK_POSE[:2]
    obstacles = []
    tries = 0
    while len(obstacles) < cfg.n_obstacles and tries < 500:
        tries += 1
        r = rng.uniform(0.12, 0.25)
        c = np.array([rng.uniform(0.8, 4.2), rng.uniform(0.5, 4.5)])
        # keep clear of start, intermediate waypoint, the docking lead-in and other obstacles
        seg = lead_end - C.LEADIN
        t = np.clip(np.dot(c - C.LEADIN, seg) / np.dot(seg, seg), 0, 1)
        d_lead = np.linalg.norm(C.LEADIN + t * seg - c)
        ok = (np.linalg.norm(c - start[:2]) > r + 0.5 and np.linalg.norm(c - C.WAYPOINT_1) > r + 0.5
              and d_lead > r + 0.45 and np.linalg.norm(c - C.LEADIN) > r + 0.6
              and all(np.linalg.norm(c - np.array(o[:2])) > r + o[2] + 0.55 for o in obstacles))
        if ok:
            obstacles.append((c[0], c[1], r))
    return start, obstacles


class World:
    def __init__(self, rng, cfg):
        self.rng, self.cfg = rng, cfg
        self.start, self.obstacles = sample_scenario(rng, cfg)
        self.x = self.start.copy()
        self.v = self.w = 0.0                                    # actual wheel-surface velocities
        self.bias = rng.normal(0, C.SIGMA_BIAS, 2) + cfg.extra_disturbance * np.array([-1.0, 1.0])
        self.collided = False
        self.t = 0.0

    def step(self, v_cmd, w_cmd, dt=C.DT):
        a = dt / (C.MOTOR_TAU + dt)                              # first-order actuator lag
        self.v += a * (v_cmd - self.v)
        self.w += a * (w_cmd - self.w)
        vl, vr = body_to_wheels(self.v, self.w)
        # encoders measure the wheel-surface speed (they do not see slip)
        odo = wheels_to_body(vl + self.rng.normal(0, C.SIGMA_ENC), vr + self.rng.normal(0, C.SIGMA_ENC))
        # ground motion: wheel slip scales each wheel's contribution
        sl = 1 + self.bias[0] + self.rng.normal(0, C.SIGMA_SLIP)
        sr = 1 + self.bias[1] + self.rng.normal(0, C.SIGMA_SLIP)
        vg, wg = wheels_to_body(vl * sl, vr * sr)
        self.x = f_motion(self.x, (vg, wg), dt)
        self.t += dt
        self._check_collision()
        return odo

    def _check_collision(self):
        x, y = self.x[:2]
        xmin, xmax, ymin, ymax = C.ROOM
        if min(x - xmin, xmax - x, y - ymin, ymax - y) < C.ROBOT_RADIUS * 0.9 and not self._in_dock_zone():
            self.collided = True
        for cx, cy, r in self.obstacles:
            if np.hypot(x - cx, y - cy) < r + C.ROBOT_RADIUS:
                self.collided = True

    def _in_dock_zone(self):
        return abs(self.x[1] - C.DOCK_MARKER[1]) < 0.3 and self.x[0] > 4.5

    def camera(self):
        return observe_markers(self.x, self.obstacles, self.rng, self.cfg.cam_fov_deg) if self.cfg.use_camera else []

    def lidar(self):
        return ray_cast(self.x, self.obstacles, self.rng)

    def truth(self):
        """GROUND TRUTH: for evaluation and plotting only. Never passed to the estimator/controller."""
        return self.x.copy()
