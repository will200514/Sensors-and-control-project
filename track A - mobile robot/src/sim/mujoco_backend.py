import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import cv2
from .. import config as C
from ..model.diffdrive import wheels_to_body, body_to_wheels
from ..perception.lidar import BEAM_ANGLES
from ..perception.aruco_detector import make_detector, detect_markers
from .world import sample_scenario

ROOM_XML = Path(__file__).resolve().parents[2] / "assets" / "room.xml"


class MujocoWorld:
    def __init__(self, rng, cfg):
        self.rng, self.cfg = rng, cfg
        self.t, self.collided = 0.0, False

        # Reproducible random start and obstacles for this seed
        self.start, self.obstacles = sample_scenario(rng, cfg)

        # Load the room (robot included). 25 physics steps of 2 ms = one 50 ms control step.
        self.m = mujoco.MjModel.from_xml_path(str(ROOM_XML))
        self.d = mujoco.MjData(self.m)
        self.n_sub = round(C.DT / self.m.opt.timestep)

        # Move this trial's obstacles into the 8 spare slots; park unused slots far away
        for i in range(8):
            body, geom = self.m.body(f"obst{i}").id, self.m.geom(f"obst{i}").id
            if i < len(self.obstacles):
                x, y, r = self.obstacles[i]
                self.m.body_pos[body] = [x, y, 0]
                self.m.geom_size[geom, 0] = r
                self.m.geom_rbound[geom] = np.hypot(r, 0.2)
            else:
                self.m.body_pos[body] = [50, 50, 0]

        # Put the robot at the start pose (x, y, heading as a quaternion)
        x, y, th = self.start
        self.d.qpos[:7] = [x, y, 0, np.cos(th / 2), 0, 0, np.sin(th / 2)]
        mujoco.mj_forward(self.m, self.d)

        # Wheel angle addresses (the encoders) and the small per-trial wheel-size error
        self.wheels = [self.m.joint("wheel_left").qposadr[0], self.m.joint("wheel_right").qposadr[0]]
        self.bias = rng.normal(0, C.SIGMA_BIAS, 2)

        self.renderer = mujoco.Renderer(self.m, 480, 640)
        self.detector = make_detector()
        # Adds overview cam
        self.viewer = None
        if cfg.viewer:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            # Start the viewer on the overview camera
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.m.camera("overview").id

    def step(self, v_cmd, w_cmd, dt=C.DT):
        # Commanded (v, w) -> left/right wheel speeds in rad/s for the two velocity motors
        vl, vr = body_to_wheels(v_cmd, w_cmd)
        self.d.ctrl[:] = [vl / C.WHEEL_RADIUS, vr / C.WHEEL_RADIUS]

        # Optional disturbance: a 6 N sideways shove for 0.3 s
        base = self.m.body("base").id
        if 0 <= self.cfg.push_time <= self.t < self.cfg.push_time + 0.3:
            th = self.truth()[2]
            self.d.xfrc_applied[base, :2] = 6.0 * np.array([-np.sin(th), np.cos(th)])
        else:
            self.d.xfrc_applied[base, :2] = 0

        # Run the physics for one control step, watching how far each wheel turned
        angle_before = self.d.qpos[self.wheels].copy()
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
        turned = self.d.qpos[self.wheels] - angle_before

        # Encoders: angle -> wheel speed (with wheel-size error and noise), then -> (v, w) for the EKF
        vl_m, vr_m = turned * C.WHEEL_RADIUS / (1 + self.bias) / dt + self.rng.normal(0, C.SIGMA_ENC, 2)

        self.t += dt
        self._check_collision()
        if self.viewer is not None and self.viewer.is_running():
            self.renderer.update_scene(self.d, camera="cam")
            image = self.renderer.render()
            vp = self.viewer.viewport
            corner = mujoco.MjrRect(vp.left + 10, vp.bottom + 10, 640, 480)
            self.viewer.set_images((corner, image))
            self.viewer.sync()
            time.sleep(dt)
        return wheels_to_body(vl_m, vr_m)

    def camera(self):
        if not self.cfg.use_camera:
            return []
        # Render what the onboard camera sees, add a little pixel noise, detect the markers
        self.renderer.update_scene(self.d, camera="cam")
        gray = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2GRAY).astype(float)
        gray = np.clip(gray + self.rng.normal(0, 2.0, gray.shape), 0, 255).astype(np.uint8)
        return detect_markers(gray, self.detector)

    def lidar(self):
        # 120 horizontal rays from the LiDAR site, against walls and obstacles (geom group 0)
        site = self.m.site("lidar").id
        origin = self.d.site_xpos[site]
        R = self.d.site_xmat[site].reshape(3, 3)              # the sensor's own orientation
        angles = np.arctan2(R[1, 0], R[0, 0]) + BEAM_ANGLES
        dirs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
        hit_id = np.zeros(len(angles), np.int32)
        dist = np.zeros(len(angles))
        mujoco.mj_multiRay(self.m, self.d, origin, dirs.ravel(), np.array([1, 0, 0, 0, 0, 0], np.uint8),
                           1, self.m.body("base").id, hit_id, dist, None, len(angles), C.LIDAR_MAX)
        # -1 means nothing hit within range
        dist = np.where(dist < 0, C.LIDAR_MAX, dist)
        return dist + self.rng.normal(0, C.LIDAR_SIGMA, len(dist)) * (dist < C.LIDAR_MAX)

    def truth(self):
        """GROUND TRUTH: for scoring and plots only, never given to the EKF or controller."""
        x, y = self.d.qpos[0:2]
        qw, qx, qy, qz = self.d.qpos[3:7]
        return np.array([x, y, np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))])

    def _check_collision(self):
        # Geometric collision check for evaluation
        x, y = self.truth()[:2]
        in_dock = abs(y - C.DOCK_MARKER[1]) < 0.3 and x > 4.5
        if min(x, 5 - x, y, 5 - y) < C.ROBOT_RADIUS * 0.9 and not in_dock:
            self.collided = True
        for cx, cy, r in self.obstacles:
            if np.hypot(x - cx, y - cy) < r + C.ROBOT_RADIUS:
                self.collided = True

    def close(self):
        """Release the live viewer and offscreen renderer."""
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()
