# Track A: single-marker TurtleBot3 docking

## Run on macOS

From a terminal with Anaconda available:

```bash
conda activate ir_env
cd "/Users/willmunro/ir-env/Sensors-and-control-project/track A - mobile robot"
python -m pip install -r requirements.txt
mjpython main.py --seed 3
```

Without the live viewer: `python main.py --seed 3 --no-view`.
The room XML is regenerated automatically. Diagnostics are saved in `results/`.
Use `python -m pytest -q` for the geometry/estimator/controller tests.

## Behaviour and coordinate frames

1. SEARCH: the EKF starts at exactly `[0, 0, 0]` in a frame defined by the initial robot pose. Initial covariance is small because that pose defines the frame, not because the robot knows a simulator coordinate. It rotates in place using encoder odometry. It cannot plan towards the dock before detection.
2. ACQUIRE: the camera detects only ArUco ID 1 (DICT_4X4_50). All four ordered corners are preserved. Square-marker PnP supplies candidate poses; an upright-marker/level-robot fit minimises their reprojection error. No marker simulation coordinates enter this calculation.
3. SWITCH: the origin becomes the floor point directly beneath the marker centre. Dock +x points out of the wall; +y runs along the wall; +z is up. The robot's heading towards the marker is approximately +/-pi in this frame. Marker pose is not GPS: this is a newly established reference frame.
4. NAVIGATE: discard the old occupancy grid, initialise pose and covariance from the acquisition measurement once, and create a new LiDAR map in dock coordinates. The planner uses marker-relative approach and stopping offsets. Later camera poses are EKF corrections; wheel odometry predicts between observations.
5. DOCK: stop within 2.5 cm of the estimated target and align within 3 degrees for 20 control steps. Final success is independently scored against simulated truth (5 cm/5 degrees and no collision).
6. REACQUIRE: after 8 seconds without an accepted fix, stop translation and rotate. Search/reacquisition times out rather than navigating forever without the marker.

The floor origin lies below the marker because the estimated state is planar `[x, y, heading]`. The marker centre itself is at height `MARKER_HEIGHT` above that origin.

## What the robot is told

Known: marker ID and physical side length, calibrated camera intrinsics/distortion, camera mounting transform, upright marker height, a level-floor assumption, and stopping/approach offsets relative to the marker.

Unknown to navigation: the simulator's absolute start pose, marker position, docking-zone position, room location and obstacle positions. These remain in the simulation for scene generation and in evaluation code for scoring/plots. Physical simulator coordinates have to exist to render the scene; they are not fed to localisation or planning.

`C.DOCK_TARGET = [0.30, 0, pi]` is a marker-relative target, not a preloaded room coordinate. `C.DOCK_APPROACH = [1.88, 0]` defines a straight final approach relative to the same marker. Other offsets can be chosen without supplying the dock's simulator location.

## Camera geometry and uncertainty

Camera axes: OpenCV x right, y down, z forward. Robot axes: x forward, y left, z up. The camera is 6 cm forward and 18 cm above the robot origin, looking forward. The 20 cm square marker is centred at 18 cm height. Both heights and the marker size were changed together so the marker is detectable from the initial 4-5 m region and its lower corners are not hidden behind the green charging-station visual. XML and pose-estimation transforms agree.

The renderer uses the existing 640x480 camera (approximately fx=530 px, principal point 320,240), with zero distortion. `fx_filter` retains the calibration-error ablation. A real camera requires actual calibration and mounting measurements; nominal simulation settings are not a substitute.

The corner-fit Jacobian maps assumed pixel noise into a full 3x3 pose covariance, preserving position-heading correlations. Nearly ambiguous candidate fits inflate that covariance. Additional calibration/model floors are added. This is a local first-order approximation, not a guarantee that a small distant planar marker is accurately localised. Far-away or nearly face-on markers can produce poor heading/lateral information; covariance must reflect that. The initial camera fix may have nonzero actual error. Changing the frame does not remove measurement error or justify reducing noise scales to make plots look better.

Tuning in `src/config.py`:

- `r_scale`: scales the full camera covariance, including the acquisition covariance.
- `corner_sigma_px`: assumed corner-coordinate standard deviation.
- `camera_pos_floor`, `camera_head_floor_deg`: model/calibration uncertainty floors.
- `q_scale`: odometry process-noise covariance multiplier.
- `gate`: full-pose NIS threshold, 11.345 for a 99% three-dimensional gate; `None` disables gating.
- `search_w`, `search_timeout`, `marker_lost_timeout`: startup/reacquisition behaviour.

The acquisition image is not immediately fused again as an independent observation. Subsequent frames are used for normal updates. Persistent calibration and encoder biases remain approximate in this three-state filter; neither is estimated explicitly.

## Mapping, control and plots

The LiDAR map supports negative dock-frame coordinates. Its finite array extent is a computational bound, not a map of the simulator room. Walls and obstacles are mapped from scans. Mapping begins after acquisition, so pre-acquisition scans are not placed in the wrong frame.

The LQR tracker uses the estimated pose only. It aligns with a path before driving if the heading discrepancy is large. Trajectory progress is based on proximity rather than a forced index increment every control call. Reaching the final trajectory index is not used as proof of docking.

The actual pose and estimate are compared in matching frames at every sample. Plots show the marker at `(0,0)` and mark the acquisition time with a green vertical line. Search-phase trajectories are transformed for display using evaluation-only information, never fed back to navigation. Docking RMSE/NEES summaries exclude the search phase; logs retain a `frame` label and both phases. The odometry-only comparison receives the same initial camera fix and no subsequent corrections.

The simulation obstacle sampler leaves an initial clear sight line to the marker, in addition to keeping the docking approach clear. This deliberately makes stationary-search startup observable in the default scenarios. It is a scene-generation constraint, not information given to the robot. If the marker is fully occluded, outside the camera's usable range or disabled, stationary rotation cannot discover it: the robot stops after the search timeout. Exploration around occluders is not implemented.

## Files changed

- `src/perception/aruco_detector.py`: one docking ID; returns all four corners.
- `src/perception/docking_pose.py` (new): camera/marker transforms, PnP, planar fit and covariance.
- `src/estimation/ekf.py`: one-time camera initialisation and full-pose Joseph updates; corrected wheel-noise cross-covariance sign.
- `src/mission.py`: search/acquire/navigate/reacquire stages; no truth-based startup or global dock target.
- `src/evaluation_frames.py` (new): explicit evaluation-only truth/display transforms.
- `src/perception/lidar.py`, `src/control/planner.py`: local-map bounds and coordinate conversion.
- `src/control/lqr_tracker.py`: proximity-based progress and initial heading alignment.
- `src/config.py`: one marker, local offsets, camera geometry, search/noise settings.
- `src/sim/world.py`: startup sight-line clearance for the simulation layout.
- `tools/make_room.py`, `assets/room.xml`: render only marker ID 1.
- `assets/robotis_tb3/turtlebot3_burger_a4.xml`: camera mounting height updated.
- `main.py`, `make_demo.py`: frame-consistent plots and acquisition diagnostics.
- `evaluate.py`: remove unsupported old configuration arguments; dock-phase NEES.
- `tests/test_docking.py`: geometry, covariance, frame transition, detector and navigation checks.

Legacy range/bearing helpers in `camera.py` remain for the original model/Jacobian tests but are not used by the single-marker EKF path.

## Evaluation

Quick held-out baseline and push trials: `python evaluate.py mujoco 1` (on a Mac, no live viewer is used). The full legacy evaluation command without arguments runs many trials and can take a long time. With `use_camera=False`, this design intentionally cannot acquire the dock; it reports `marker_not_found` rather than using a hidden ground-truth initialisation.

See `VALIDATION.md` for checks actually run for this version.
