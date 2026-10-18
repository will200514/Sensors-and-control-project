# Track A: MuJoCo navigation and docking

## Run this complete version

Extract this folder beside your current project and open it in VS Code. All four robot STL meshes and the original asset licence/source notice are included. You do not need to merge files or install new packages if your existing `ir_env` already runs the original project.

With `ir_env` active, change into the extracted folder and run:

```bash
mjpython main.py --seed 3
```

The viewer opens and diagnostic plots are saved under `results/`. Close the viewer to end a run early and plot samples collected so far. A run with no samples prints a message instead of plotting empty arrays.

For a run without the live viewer (still MuJoCo physics and rendered camera):

```bash
mjpython main.py --seed 3 --no-view
```

For an animated path/error GIF plus the diagnostic PNG:

```bash
mjpython make_demo.py --seed 3
```

The animation runs MuJoCo without a live viewer while collecting results. Optional disturbance: add `--push 20` for the existing sideways push at 20 seconds.

Both entry points regenerate `assets/room.xml` from configuration before every run. Manual edits to that generated XML will be overwritten: edit `tools/make_room.py` or `src/config.py` instead.

## What changed

- Removed the intermediate waypoint and its mission phase. Navigate to `LEADIN`, then the existing straight docking approach. LEADIN remains necessary.
- Exactly two ArUco markers: ID 0 at (4.98, 2.0), ID 1 at (4.98, 0.8), both facing -X. ID 1 is the docking marker. Configuration and generated geometry use the same ordering.
- Removed the NumPy backend selector, synthetic camera observations and synthetic LiDAR ray casting. `world.py` now only generates random starts/obstacles.
- Removed unused NumPy-simulator tuning options. Kept NumPy as a numerical dependency for EKF, planning, control and plots.
- Preserved estimator/controller tuning and wheel physics; removed duplicated imports and viewer updates; added cleanup of renderer/viewer.
- Fixed a planner array-shape error when clearing start/goal cells near map boundaries.
- Added automatic room regeneration to prevent stale marker geometry after configuration edits.
- Added no-sample plot handling and corrected animation timing to match logged simulation time.

## Diagnostic plots

1. Actual path, EKF estimate and wheel-odometry-only baseline, with 2-sigma covariance ellipses.
2. Position error plus `2 sqrt(Pxx+Pyy)` as an uncertainty scale (not an exact 95% radial confidence bound).
3. Signed heading error with a +/-2-sigma band.
4. Distance from actual position to the active reference point (not nearest-path cross-track distance).
5. NEES for the three-state filter; expected value 3 and individual-sample 95th percentile 7.815 under ideal Gaussian assumptions. This line is not a confidence bound on mean NEES.
6. Number of detections in the latest camera frame, held between frames.

Dead reckoning is a useful baseline derived from MuJoCo wheel measurements, not a second simulator. Ground truth is used for initialisation/evaluation, not supplied to the tracking controller.

Two markers provide less coverage than five, especially on the opposite side of the room. Drift and success rate can change. Marker positions are a starting layout, not a verified optimum. This update does not claim successful docking for all seeds.

## Files retained and optional old files

`src/config.py`, `mission.py`, the control/model/estimation modules, the camera measurement model, ArUco detector, occupancy mapper, scenario generator, and MuJoCo backend remain.

The uploaded `src/perception/calibration.py` generates synthetic calibration observations; it is not required by these entry points and is omitted from this bundle. The old `assets/calib.xml` and separate calibration script are also omitted from this focused runnable folder. Your original ZIP remains unchanged. External scripts that use removed synthetic functions or `Cfg(backend=...)` need updating; only the supplied entry points are covered by this cleanup.

## Validation performed

- All delivered Python files compile and entry-point imports succeed.
- Every referenced `C.<setting>` exists in the delivered configuration.
- Generated room contains exactly two named marker backings, IDs 0 and 1.
- OpenCV detects both generated marker patterns with the selected dictionary.
- Scenario generation is reproducible for seed 3 and returns four obstacles.
- Planning works for a start close to the room boundary.
- Six-panel plotting ran with synthetic smoke-test data; output was visually checked. That data is not included as simulation results.
- Full seed-3 headless MuJoCo run completed using all original robot meshes: no collision, successful docking, approximately 0.0111 m dock position error and 0.0351 rad heading error (MuJoCo 3.16.0).
- Motion/camera Jacobian checks and controllability tests from the original project passed.
- The GIF/plot entry point was also checked against the project's pinned MuJoCo 3.10.0; see VALIDATION.md for the final result.
- Linux headless validation does not verify the macOS live viewer. Other seeds and disturbance scenarios have not been evaluated.
- Seed 3 mean NEES was about 13.17 in the initial full run, above the ideal value 3: the filter can be overconfident. No estimator tuning was changed or results hidden.

## Scope of the complete folder

This is a focused runtime, not a copy of every historical experiment. Old results, caches, `analysis.py`, `tune.py`, `evaluate.py`, `diagnose_lqr.py`, the standalone keyboard experiment and synthetic calibration tools are omitted. The numerical model tests relevant to this version remain in `tests/test_models.py`. Their removed test covered synthetic calibration only. The main runner and animation runner are both MuJoCo-only.
