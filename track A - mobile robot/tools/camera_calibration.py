"""
Introusive calibration (internal mesured by robot) not extrusive: camera offset, lidar distance on robot/ external truths.

Calibrate the robot's MuJoCo camera with a checkerboard (Part B):  python tools/calibrate_camera.py

Shows the camera the board in 30 random poses, finds the corners, runs cv2.calibrateCamera,
and compares the result with the true focal length set by the camera's fovy.
Corner finding and the 9 x 6 / 25 mm board are from the Week 3 tutorial (measured_corners, Part C).
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

gx, gy, sq = 9, 6, 0.025
# Board corners in the board's own frame (Z = 0 plane), as in the Week 3 tutorial
P_board = np.array([[i * sq, j * sq, 0] for j in range(gy) for i in range(gx)], np.float32)


def measured_corners(g):
    """Sub-pixel checkerboard corners (Week 3 tutorial)."""
    ok, c = cv2.findChessboardCorners(g, (gx, gy))
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)


m = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "calib.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
cam = d.cam_xpos[m.camera("cam").id].copy()
r = mujoco.Renderer(m, 480, 640)
rng = np.random.default_rng(0)
obj, img = [], []
save = "--save" in sys.argv
if save:
    (C.RESULTS_DIR / "calib_images").mkdir(exist_ok=True)

while len(img) < 30:
    # Random board pose in front of the camera, spread across the whole image
    dist = rng.uniform(0.3, 0.7)
    d.mocap_pos[0] = cam + [dist, rng.uniform(-0.45, 0.45) * dist, rng.uniform(-0.3, 0.3) * dist]
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [rng.uniform(-0.35, 0.35), rng.uniform(-0.5, 0.5), np.pi + rng.uniform(-0.5, 0.5)], "xyz")
    d.mocap_quat[0] = q
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera="cam")
    g = cv2.cvtColor(r.render(), cv2.COLOR_RGB2GRAY)
    c = measured_corners(g)
    if c is not None:
        obj.append(P_board); img.append(c)
        if save:
            cv2.imwrite(str(C.RESULTS_DIR / "calib_images" / f"view_{len(img):02d}.png"),
                        cv2.drawChessboardCorners(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), (gx, gy), c, True))

rms, K, dist_coef, _, _ = cv2.calibrateCamera(obj, img, (640, 480), None, None)
fx_true = 240 / np.tan(np.deg2rad(m.cam_fovy[m.camera("cam").id]) / 2)
print(f"{len(img)} views, RMS reprojection error {rms:.3f} px")
print(f"fx {K[0, 0]:.1f} px (true {fx_true:.1f}, config {C.FX})   fy {K[1, 1]:.1f}   cx {K[0, 2]:.1f}   cy {K[1, 2]:.1f}")
print(f"fx error {100 * (K[0, 0] - fx_true) / fx_true:+.2f} %")