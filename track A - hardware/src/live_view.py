"""[NEW] Live window for hardware runs.

Left:  camera image with the ArUco detection drawn on it.
Right: top-down map in the dock frame (marker at the origin, x out from the wall):
       occupancy map, inflated no-go area, LiDAR points, planned route,
       approach point, dock target and robot pose.

Press q or Esc in the window to stop the robot and end the run.
If OpenCV cannot open a window (headless build), frames are saved to
results/live_view.jpg once a second instead.
"""
import time

import cv2
import numpy as np

from . import config as C
from .model.diffdrive import wrap

MAP_PX = 480
VIEW_X = (-0.5, 4.5)       # dock-frame view [m]; marker wall at x = 0
VIEW_Y = (-2.5, 2.5)
DRAW_PERIOD = 0.2          # redraw at 5 Hz so the control loop is not slowed

# BGR colours
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (170, 170, 170)
GREEN, RED, BLUE = (60, 170, 60), (40, 40, 220), (220, 120, 30)
ORANGE, PURPLE, YELLOW = (0, 140, 255), (160, 60, 160), (0, 210, 230)


class LiveView:
    def __init__(self, enabled=True, record=False):
        self.enabled = enabled
        self.window = enabled
        self.camera = np.zeros((480, 640, 3), np.uint8)
        self.camera_note = "waiting for camera"
        self.next_draw = 0.0
        self.next_save = 0.0
        self.writer = None
        self.record = record
        self.res = (VIEW_X[1] - VIEW_X[0]) / MAP_PX
        self.cells = None
        if enabled:
            try:
                cv2.namedWindow("Track A live view", cv2.WINDOW_NORMAL)
            except cv2.error:
                self.window = False
                print("Live view: no window support in this OpenCV build, "
                      "saving results/live_view.jpg instead.", flush=True)

    # ---------- camera panel ----------
    def camera_frame(self, image, detections, observation):
        if not self.enabled:
            return
        img = image.copy()
        if detections:
            corners = [c.reshape(1, 4, 2).astype(np.float32) for _, c in detections]
            ids = np.array([[i] for i, _ in detections], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
            for _, c in detections:
                cv2.circle(img, tuple(int(v) for v in c[0]), 5, (0, 0, 255), -1)
            if observation is not None:
                d = float(np.hypot(*observation.pose[:2]))
                self.camera_note = (f"ID {C.DOCK_MARKER_ID} ACCEPTED  {d:.2f} m  "
                                    f"side {np.mean(np.linalg.norm(np.roll(detections[0][1], -1, 0) - detections[0][1], axis=1)):.0f} px")
                colour = (0, 200, 0)
            else:
                self.camera_note = f"ID {C.DOCK_MARKER_ID} seen, pose REJECTED"
                colour = (0, 140, 255)
        else:
            self.camera_note = "no marker"
            colour = (0, 0, 255)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, self.camera_note, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        if img.shape[:2] != (480, 640):
            img = cv2.resize(img, (640, 480))
        self.camera = img

    # ---------- map panel ----------
    def _px(self, x, y):
        """Dock-frame metres -> map pixel (x up the screen, y to the left)."""
        u = (VIEW_Y[1] - np.asarray(y)) / self.res
        v = (VIEW_X[1] - np.asarray(x)) / self.res
        return u, v

    def _pt(self, x, y):
        u, v = self._px(x, y)
        return int(round(float(u))), int(round(float(v)))

    def _grid_layers(self, grid, inflate):
        if self.cells is None:
            v, u = np.mgrid[0:MAP_PX, 0:MAP_PX]
            x = VIEW_X[1] - (v + 0.5) * self.res
            y = VIEW_Y[1] - (u + 0.5) * self.res
            self.cells = grid.to_cell(x, y)
        occ = grid.L[self.cells] > 0.5
        free = grid.L[self.cells] < -0.5
        k = max(1, int(round(inflate / self.res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        blocked = cv2.dilate(occ.astype(np.uint8), kernel) > 0
        return occ, free, blocked

    def draw(self, *, elapsed, phase, pose, localised, grid, inflate, traj, idx,
             scan, forward, turning, front_min=None, note=""):
        """Redraw at most every DRAW_PERIOD. Returns True if the user asked to stop."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if now < self.next_draw:
            return False
        self.next_draw = now + DRAW_PERIOD

        m = np.full((MAP_PX, MAP_PX, 3), 235, np.uint8)
        # Before the marker is found the map frame is not defined: show the robot
        # at a fixed spot with LiDAR around it.
        if not localised:
            pose = np.array([2.0, 0.0, np.pi])
        else:
            occ, free, blocked = self._grid_layers(grid, inflate)
            m[free] = (250, 250, 250)
            m[blocked & ~occ] = (205, 205, 245)
            m[occ] = (60, 60, 60)

        # 0.5 m grid lines
        for gx in np.arange(np.ceil(VIEW_X[0] * 2) / 2, VIEW_X[1], 0.5):
            u0, v0 = self._pt(gx, VIEW_Y[0]); u1, v1 = self._pt(gx, VIEW_Y[1])
            cv2.line(m, (u0, v0), (u1, v1), (215, 215, 215), 1)
        for gy in np.arange(np.ceil(VIEW_Y[0] * 2) / 2, VIEW_Y[1], 0.5):
            u0, v0 = self._pt(VIEW_X[0], gy); u1, v1 = self._pt(VIEW_X[1], gy)
            cv2.line(m, (u0, v0), (u1, v1), (215, 215, 215), 1)

        if localised:
            # marker on the wall, approach point, dock target, docking line
            h = C.MARKER_L / 2
            cv2.line(m, self._pt(0, -h), self._pt(0, h), PURPLE, 4)
            cv2.putText(m, "marker", self._pt(-0.12, 0.3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, PURPLE, 1)
            cv2.line(m, self._pt(*C.DOCK_APPROACH[:2]), self._pt(*C.DOCK_TARGET[:2]), YELLOW, 1)
            cv2.circle(m, self._pt(*C.DOCK_APPROACH[:2]), 5, YELLOW, 2)
            cv2.drawMarker(m, self._pt(*C.DOCK_TARGET[:2]), PURPLE, cv2.MARKER_CROSS, 14, 2)

            # planned route: done part grey, remaining part green
            if traj is not None and len(traj) > 1:
                u, v = self._px(traj[:, 0], traj[:, 1])
                pts = np.stack([u, v], 1).round().astype(np.int32)
                i = int(min(max(idx, 0), len(pts) - 1))
                if i > 0:
                    cv2.polylines(m, [pts[:i + 1]], False, GREY, 2)
                cv2.polylines(m, [pts[i:]], False, GREEN, 2)

        # LiDAR points
        if scan is not None:
            a = np.asarray(scan.angles, float); r = np.asarray(scan.ranges, float)
            ok = np.isfinite(a) & np.isfinite(r) & (r >= scan.range_min) & (r <= min(scan.range_max, C.LIDAR_MAX))
            a, r = a[ok] + getattr(C, "LIDAR_YAW", 0.0), r[ok]
            th = pose[2]
            ox, oy = getattr(C, "LIDAR_OFFSET_X", 0.0), getattr(C, "LIDAR_OFFSET_Y", 0.0)
            sx = pose[0] + np.cos(th) * ox - np.sin(th) * oy
            sy = pose[1] + np.sin(th) * ox + np.cos(th) * oy
            u, v = self._px(sx + r * np.cos(th + a), sy + r * np.sin(th + a))
            inside = (u >= 0) & (u < MAP_PX) & (v >= 0) & (v < MAP_PX)
            for uu, vv in zip(u[inside].astype(int), v[inside].astype(int)):
                cv2.circle(m, (uu, vv), 2, BLUE, -1)

        # robot, heading and the 0.26 m front stop zone (+-35 deg)
        c = self._pt(pose[0], pose[1])
        rad = max(2, int(C.ROBOT_RADIUS / self.res))
        stop = max(0.20, C.ROBOT_RADIUS + 0.06)
        arc = [self._pt(pose[0] + stop * np.cos(pose[2] + t), pose[1] + stop * np.sin(pose[2] + t))
               for t in np.linspace(-np.deg2rad(35), np.deg2rad(35), 12)]
        cv2.polylines(m, [np.array([c] + arc + [c], np.int32)], False, RED, 1)
        cv2.circle(m, c, rad, ORANGE, 2)
        cv2.line(m, c, self._pt(pose[0] + 1.6 * C.ROBOT_RADIUS * np.cos(pose[2]),
                                pose[1] + 1.6 * C.ROBOT_RADIUS * np.sin(pose[2])), ORANGE, 2)

        # status text
        lines = [f"t {elapsed:5.1f} s   {phase}",
                 f"v {forward:+.3f} m/s   w {turning:+.3f} rad/s",
                 (f"pose x {pose[0]:.2f}  y {pose[1]:+.2f}  th {np.degrees(wrap(pose[2])):+.0f} deg"
                  if localised else "not localised: LiDAR around robot only")]
        if front_min is not None and np.isfinite(front_min):
            lines.append(f"front gap {front_min:.2f} m")
        if note:
            lines.append(note)
        box = m[0:12 + 18 * len(lines), 0:330]
        box[:] = (0.25 * box + 0.75 * 255).astype(np.uint8)
        for k, text in enumerate(lines):
            cv2.putText(m, text, (8, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
        cv2.putText(m, "q / Esc = stop", (MAP_PX - 120, MAP_PX - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)

        frame = np.hstack([self.camera, m])

        if self.record and self.writer is None:
            path = str(C.RESULTS_DIR / "live_view.mp4")
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          1 / DRAW_PERIOD, (frame.shape[1], frame.shape[0]))
            print("Recording live view to", path, flush=True)
        if self.writer is not None:
            self.writer.write(frame)

        if self.window:
            try:
                cv2.imshow("Track A live view", frame)
                key = cv2.waitKey(1) & 0xFF
                return key in (ord("q"), 27)
            except cv2.error:
                self.window = False
        if now >= self.next_save:
            cv2.imwrite(str(C.RESULTS_DIR / "live_view.jpg"), frame)
            self.next_save = now + 1.0
        return False

    def close(self, hold=True):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.window:
            try:
                if hold:
                    print("Run finished. Press any key in the live view window to close it.", flush=True)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except cv2.error:
                pass
