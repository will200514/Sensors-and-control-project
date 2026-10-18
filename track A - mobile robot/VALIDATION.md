# Validation

The complete supplied robot meshes were used for headless Linux validation.

- MuJoCo 3.10.0 (the project pin): `make_demo.py --seed 3` completed, docked successfully, and produced the included diagnostic PNG and 166-frame GIF. Dock position error was approximately 0.011 m.
- MuJoCo 3.16.0: `main.py --seed 3 --no-view` completed with no collision; dock position error 0.0111 m, heading error 0.0351 rad, simulation time 33.15 s.
- Original numerical checks for motion Jacobians, camera Jacobian and controllability passed by invoking their test functions directly. The pytest runner was not installed in the validation environment.
- Python compilation, entry-point imports, configuration references, marker IDs/pattern detection and boundary planning checks passed.
- The delivered PNG was visually inspected. Results are actual MuJoCo simulation output.

The macOS live viewer, other seeds and disturbance scenarios were not tested. The EKF remains overconfident during parts of the run (mean NEES about 13.17 in the initial full run, versus ideal 3). Existing tuning was preserved; successful docking on this seed is not a guarantee for all scenarios.
