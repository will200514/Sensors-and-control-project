# Validation of the single-marker version

Executed in a Linux headless MuJoCo renderer (`MUJOCO_GL=egl`), with Python 3.12, MuJoCo 3.10.0 and OpenCV 4.14. The macOS live-view command remains `mjpython main.py`; a macOS GUI was not available here.

## Automated checks

`python -m pytest -q`: 14 tests passed. Checks cover motion and legacy camera Jacobians, controllability, full corner-to-pose round trips including fronto-parallel views, positive camera covariance and r_scale, tiny-marker rejection, one-time EKF frame initialisation, full-pose update, wheel covariance sign, coordinate-frame round trips, negative map coordinates, stationary trajectory progress, independence of planning from simulator dock coordinates, and detection of the docking ID only.

`python -m compileall -q src main.py tools make_demo.py evaluate.py`: passed.

## Rendered end-to-end smoke checks

| Scenario | Result | Acquisition | Total simulation time | Final position error | Final heading error | Collision |
|---|---|---|---|---|---|---|
| Seed 3, default four obstacles | Docked | 17.0 s | 46.5 s | 2.94 cm | 1.23 degrees | No |
| Held-out seed 1000, default four obstacles | Docked | 16.6 s | 51.3 s | 3.03 cm | 0.58 degrees | No |
| Seed 3, camera disabled, search timeout shortened to 0.3 s | marker_not_found | None | 0.3 s | Not applicable | Not applicable | No |

The disabled-camera check confirms that the robot does not quietly fall back to a known simulator dock position. Default search timeout is 40 seconds, not 0.3 seconds.

Plots for the two successful runs are included in `results/`. They were inspected for matching coordinate frames and acquisition markers.

## Limits of these checks

These are smoke checks, not a statistical robustness study. The full legacy evaluation sweep was not run. GIF generation and the live macOS viewer were not exercised.

Full-pose estimation from a small, distant planar marker is noisy and can be ambiguous. Seed 3 had dock-phase position RMSE 0.547 m (including its initial noisy camera fix), reducing to the final 0.0294 m docking error; seed 1000 had RMSE 0.196 m. These results have not been hidden by resetting truth or tuning down the plotted error.

Dock-phase mean NEES was 8.60 and 7.38 respectively, above the ideal three-state expectation of 3. This implementation does not establish statistical consistency of the covariance model. Rasterisation, planar-pose ambiguity, calibration assumptions and persistent wheel bias can contribute. Noise floors/scales remain explicit tunables; further calibration/evaluation is needed before claiming consistency.

The scene generator deliberately keeps an initial line of sight to the marker. Fully occluded-marker exploration is not implemented; stationary search then times out. The 20 cm marker and camera are both centred at 18 cm height to avoid occlusion by the charging-station visual and to fit the marker into the close-range field of view.
