# Track A: TurtleBot3 visual waypoint navigation and docking

**Status: DRAFT starter project.** The full pipeline (model, sensing, EKF, LQR tracker, LiDAR planning, docking,
evaluation) runs end to end, but on a **pure-NumPy stand-in simulator, not MuJoCo** (MuJoCo was not available where this
was written). See "What is still yours to do". Nothing in this README claims MuJoCo results.

## 1. Task and pipeline

A TurtleBot3 Burger starts at a random pose in a 5 m x 5 m room with 4 random cylindrical obstacles, drives through
an intermediate visual waypoint, and docks with an ArUco-marked charging station on the right wall.

```
wheel encoders --> EKF predict --\
                                  +--> x_hat, P --> LQR tracker --> (v, w) --> robot
ArUco range/bearing --> EKF update/                      ^
LiDAR + x_hat --> occupancy grid --> A* --> trajectory --/
```
The controller and planner use **only the EKF estimate**. The simulator's true pose is read in exactly two
places: `World.truth()` inside the evaluation/logging block of `src/mission.py` and the plotting scripts.

Run: `pip install -r requirements.txt`, then `python main.py --seed 3` (one mission + plots in `results/`),
`python evaluate.py` (50-trial baseline + ablations, ~1 min), `python analysis.py`, `python make_demo.py --seed 3`,
`python tune.py`, `python diagnose_lqr.py`, `python -m pytest tests` (or call the test functions directly).

## 2. Part A: state-space model

State x = [x, y, theta] (world frame), input u = [v, omega] from the wheel encoders, v = (vR+vL)/2, omega = (vR-vL)/b,
b = 0.160 m. Discrete exact-arc model over dt = 0.05 s (Week 7/8 model rewritten without the omega = 0 singularity):
`x+ = x + v dt sinc(w dt/2) cos(theta + w dt/2)`, same with sin for y, `theta+ = theta + w dt`.
Jacobians F = df/dx and G = df/du are in `src/model/diffdrive.py` and checked against finite differences in `tests/`.

Assumptions: planar motion, no lateral slip in the model (slip enters as noise), wheel speeds follow commands instantly in
the model (the simulator has a 0.12 s motor lag and per-wheel slip bias that the model does *not* know about).

**Tracking error model (for the LQR).** With e = [R(theta)^T (p_ref - p), theta_ref - theta] and u = u_ref + du,
linearising about e = 0 gives `de/dt = A e + B du` with A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],
B = [[-1, 0], [0, 0], [0, -1]]. Discretised exactly (ZOH) at each reference (v_r, w_r).

**Controllability / observability** (`python analysis.py`):
```
CONTROLLABILITY of the error model (A(v_r, w_r), B)  [rank must be 3]
   v_r   w_r  rank   sigma_min(ctrb)
  0.18  0.00     3            0.1800
  0.18  0.50     3            0.5314
  0.04  0.00     3            0.0400
  0.01  0.00     3            0.0100
  0.00  0.50     3            0.5000
  0.00  0.00     2            0.0000
-> at v_r = 0 and w_r = 0 the lateral error e_y cannot be corrected (rank 2): a robot must move to fix a sideways offset.
   With v_r = 0, w_r != 0 the model is controllable, but only because the reference itself is rotating.

OBSERVABILITY of the EKF linearisation over a 1 s window (20 steps), pose (1, 1, 0)
case                             rank  singular values
no marker visible, moving           0  [0. 0. 0.]
1 marker ahead, stationary          2  [5.1546 4.4721 0.    ]
1 marker ahead, moving              3  [5.3184 4.4721 0.0432]
1 marker off-axis, moving           3  [4.9909 4.3927 0.0481]
2 markers, stationary               3  [7.1149 4.9689 4.0361]
2 markers, moving                   3  [7.3142 4.9681 3.9297]
-> one marker gives range+bearing (2 equations) for 3 states: not instantaneously observable. Heading is recovered
   only through motion (position-heading coupling in F) or a second marker. With no marker, only the predicted
   covariance exists: P grows without bound (see dead-reckoning error in results/).
```

## 3. Part B: sensing, calibration, frames

Frames: world W; robot B (x forward, origin at axle centre); camera C, mounted 0.06 m ahead of B on its x axis, looking along
+x_B. The measurement model includes that offset: `h(x) = [|m - p_cam|, atan2(m - p_cam) - theta]` with
p_cam = p + 0.06 [cos theta, sin theta]; the Jacobian (including d/d theta from the offset) is tested numerically.

Marker detection gives a pixel column u and apparent side length s: bearing = atan((cx - u)/fx), range = fx L / s
(fx = 530 px, cx = 320 px, L = 0.10 m). **R is derived, not guessed:** sigma_rho = rho^2 sigma_s / (fx L)
(grows with range squared: 0.5 cm at 0.5 m, 7.5 cm at 2 m, 15 cm at 2.8 m for 1 px size noise), sigma_beta^2 = (sigma_u/fx)^2 + (0.5 deg)^2.
Detections are gated by FOV (62.2 deg), range (0.15 to 3.5 m), incidence angle (< 60 deg) and LiDAR-obstacle occlusion.

**Calibration** (`src/perception/calibration.py`): least squares for fx and cx from 21 views of a marker at known poses:
fx = 528.6 +/- 0.7 px from the bearing equation (531.1 from the size equation), cx = 320.3 +/- 0.2 px (true values 530, 320).
A 5% error in fx is tested as an ablation below.

LiDAR: 120 beams, 3.5 m, 1 cm noise, log-odds occupancy grid (5 cm cells) built **with the estimated pose**.

## 4. Part C: estimation (EKF)

Prediction uses odometry. Process noise Q = G Sigma_u G^T with Sigma_u derived from wheel-level noise (encoder 3 mm/s per wheel,
2% random slip, 1% slip bias folded in), see `odom_covariance`. Update: sequential range-bearing updates per detected marker,
bearing innovation wrapped, Joseph-form covariance update, chi-square(2) 99% gate (9.21), and a lock-out recovery
(if every detection in 5 consecutive frames is rejected, P is inflated so the gate re-opens).
Initial estimate = true start + N(0, diag(0.1 m, 0.1 m, 5 deg)^2), P0 matches.

**Tuning** (on tuning seeds 0-19, never on the evaluation seeds). Two scale factors multiply the derived Q and R, because
the derived values ignore the unmodelled wheel bias, motor lag, linearisation error and the 1/s nonlinearity of range.
Criterion: median NEES near the chi-square(3) median of 2.37 and a high fraction inside the 95% bound (7.81), preferring the
conservative side (Week 7: "when in doubt, inflate Q"). Chosen: **q_scale = 4, r_scale = 2**.
```
 q_scale r_scale   succ    NEES  frac<7.81  rmse[cm]  head[deg]
     0.5     0.5   0.85   11.26       0.41       6.6       2.98
     0.5     1.0   0.95    6.53       0.57       5.1       2.00
     0.5     2.0   1.00    5.43       0.61       4.9       1.91
     0.5     4.0   1.00    5.25       0.62       4.9       1.95
     1.0     0.5   0.95    7.89       0.50       5.4       1.99
     1.0     1.0   0.95    4.81       0.68       4.7       1.83
     1.0     2.0   0.95    3.83       0.78       4.7       1.81
     1.0     4.0   1.00    2.98       0.84       4.8       1.84
     2.0     0.5   0.95    5.80       0.63       4.8       1.87
     2.0     1.0   1.00    3.25       0.76       4.5       1.81
     2.0     2.0   1.00    2.39       0.84       4.4       1.76
     2.0     4.0   1.00    1.94       0.92       4.5       1.79
     4.0     0.5   0.95    4.93       0.65       4.8       1.84
     4.0     1.0   0.95    2.60       0.85       4.5       1.80
     4.0     2.0   1.00    1.88       0.91       4.2       1.74
     4.0     4.0   0.95    1.41       0.93       4.3       1.73
     8.0     0.5   0.95    3.42       0.80       4.5       1.80
     8.0     1.0   1.00    2.01       0.90       4.3       1.76
     8.0     2.0   1.00    1.25       0.95       4.1       1.71
     8.0     4.0   0.95    1.01       0.98       4.2       1.68
```

## 5. Part D: control

Time-varying LQR on the error model above, evaluated at the current reference (v_r, w_r) (gains cached on a grid).
**Weights by Bryson's rule**: Q = diag(1/0.05^2, 1/0.05^2, 1/0.1^2) (5 cm, 5 cm, 0.1 rad acceptable error),
R = diag(1/0.05^2, 1/0.5^2) (5 cm/s, 0.5 rad/s acceptable correction). Commands saturate at 0.20 m/s and 1.5 rad/s.
The tracker follows a virtual target on the reference (nearest sample + 1) using the estimated pose.

Reference generation: A* on the inflated LiDAR map (inflation 0.24 m = robot radius 0.105 + about 2 sigma of position error + margin),
line-of-sight shortcutting, Chaikin smoothing, speed profile (0.18 m/s cruise, decelerating to 0.03 m/s at the dock).
Replanning only if LiDAR shows the remaining path is blocked (or every 5 s). The last 1.6 m is a straight lead-in along the
dock marker normal; the plan is locked once the robot is on it, so the final approach is a consistent line.

## 6. Part E: results (held-out seeds 1000+, ground truth used only to score)

Success = docked with true position error < 5 cm, heading error < 5 deg, no collision, within 180 s.

## Baseline and ablations (held-out seeds)

| config | n | success [95% CI] | dock pos med/max [cm] | dock head med [deg] | time [s] | est. RMSE [cm] | dead-reck. RMSE [cm] | NEES median | frac NEES<7.81 | track RMSE [cm] | collisions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (submitted design) | 50 | 100% [93, 100] | 0.9 / 3.5 | 0.4 | 33 | 5.5 | 65.9 | 2.23 | 0.92 | 8.3 | 0 |
| no camera (odometry only) | 30 | 0% [-0, 11] | 128.9 / 320.3 | 26.7 | 33 | 68.5 | 68.5 | 4.91 | 0.58 | 68.7 | 19 |
| process noise Q x0.1 | 30 | 100% [89, 100] | 1.0 / 3.7 | 0.3 | 33 | 6.9 | 68.2 | 6.15 | 0.59 | 9.4 | 0 |
| process noise Q x10 | 30 | 100% [89, 100] | 0.9 / 2.8 | 0.4 | 33 | 5.6 | 68.9 | 1.00 | 1.00 | 8.2 | 0 |
| camera noise R x0.1 | 30 | 100% [89, 100] | 0.9 / 1.6 | 0.7 | 33 | 6.7 | 68.5 | 10.43 | 0.42 | 8.6 | 0 |
| camera noise R x10 | 30 | 100% [89, 100] | 0.9 / 3.2 | 0.4 | 33 | 6.6 | 68.0 | 1.39 | 0.96 | 8.9 | 0 |
| innovation gate off | 30 | 100% [89, 100] | 0.9 / 3.5 | 0.4 | 33 | 6.0 | 69.9 | 2.32 | 0.93 | 8.5 | 0 |
| calibration error fx +5% | 30 | 97% [83, 99] | 2.4 / 2.8 | 2.0 | 33 | 10.2 | 67.3 | 156.39 | 0.13 | 11.7 | 0 |
| calibration error fx -5% | 30 | 100% [89, 100] | 1.2 / 2.5 | 1.9 | 33 | 10.5 | 69.1 | 205.00 | 0.15 | 11.7 | 0 |
| narrow camera FOV (30 deg) | 30 | 100% [89, 100] | 0.9 / 3.4 | 0.4 | 33 | 7.8 | 68.7 | 2.55 | 0.90 | 10.3 | 0 |
| wheel slip bias +/-5% (stress) | 30 | 90% [74, 97] | 1.9 / 4.5 | 1.5 | 34 | 11.1 | 366.4 | 8.53 | 0.46 | 15.1 | 1 |
| 7 obstacles (stress) | 30 | 100% [89, 100] | 0.9 / 1.7 | 0.3 | 33 | 5.3 | 61.4 | 2.20 | 0.90 | 7.1 | 0 |

## Failure diagnosis (seed: cause)

- **no camera (odometry only)**: 1000: collision, 1001: dock tolerance, 1002: dock tolerance, 1003: collision, 1004: collision, 1005: dock tolerance, 1006: collision, 1007: collision, 1008: collision, 1009: collision, 1010: collision, 1011: collision, 1012: dock tolerance, 1013: dock tolerance, 1014: estimator lost, 1015: collision, 1016: dock tolerance, 1017: dock tolerance, 1018: collision, 1019: collision, 1020: collision, 1021: collision, 1022: collision, 1023: collision, 1024: dock tolerance, 1025: estimator lost, 1026: collision, 1027: dock tolerance, 1028: collision, 1029: collision
- **calibration error fx +5%**: 1022: dock tolerance
- **wheel slip bias +/-5% (stress)**: 1003: dock tolerance, 1009: dock tolerance, 1011: collision

## LQR weight trade-off (R scaled; Q fixed)

| R scale | success | track RMSE [cm] | control effort mean(v^2+w^2) |
|---|---|---|---|
| 0.01 | 100% | 8.26 | 0.3878 |
| 0.1 | 100% | 8.22 | 0.3304 |
| 1.0 | 100% | 8.57 | 0.2600 |
| 10.0 | 100% | 8.88 | 0.1817 |
| 100.0 | 83% | 11.98 | 0.1285 |

![evaluation](results/evaluation.png)
![lqr](results/lqr_tradeoff.png)
![run](results/run_seed3.png)

### What the numbers say (and do not say)

* **The camera is essential.** Odometry only: 0% success, 66 to 69 cm average error, most runs collide. With the camera:
  100% success (95% CI 93 to 100% over 50 trials), median docking error 0.9 cm. Docking error is small because the dock marker is
  only 0.24 m from the camera at the end, where sigma_rho is about 0.1 cm.
* **Consistency.** At the chosen tuning the median NEES (2.23) matches the ideal 2.37 and 92% of samples are inside the 95% bound
  (not 95%: outages with no marker in view make the tails heavy). Q x0.1 gives median NEES 6.2 (overconfident);
  R x0.1 gives 10.4 (overconfident: it believes the camera too much). Q x10 and R x10 are conservative (NEES 1.0, 1.4) at almost
  no cost in accuracy here, but they would respond more slowly if the odometry were truly bad.
* **Calibration matters for honesty, less for success.** A +/-5% error in fx leaves success at 97 to 100% but destroys consistency
  (median NEES 156 to 205): the filter is confident and wrong, mostly in range, and the estimated error roughly doubles (10 cm vs 5.5 cm).
* **Gate.** Turning the gate off made no measurable difference in these trials (the simulator has no outliers). Its value is untested here
  and it needed the lock-out recovery: before adding that, 2 of 10 tuning trials were lost (see Failures).
* **LQR trade-off.** R x0.01 to x10 changes tracking RMSE by only 8.3 to 8.9 cm (dominated by the start-up transient, since the robot
  starts at a random heading and the reference begins immediately) while effort falls 2x. R x100 starts to fail (83%, 5 of 30). `python diagnose_lqr.py` (output in `results/lqr_diagnosis.txt`) shows all 5 are final
  docking misses (5.1 to 21.6 cm, none are collisions). The earlier hypothesis of a weak initial turn is only partly supported: the peak tracking error
  is early (2 to 3 s, 32 to 39 cm) in 3 of the 5, but in seed 1018 the estimate is good (1.2 cm RMSE) and the dock still misses by 10.6 cm, which points
  at weak final-approach correction. Seed 1022 also has a poor estimate (21.5 cm RMSE), so it is estimator-driven. This is observational, not a controlled isolation. The limit on aggressive gains is the actuator lag (0.12 s) and saturation,
  which the model ignores.
* **Stress.** +/-5% wheel slip bias: 90% success (3 failures: 2 missed the dock tolerance, 1 collision), estimator RMSE doubles.

### Failures diagnosed during development
1. *Estimator lock-out (seeds 1 and 2 of the tuning set).* A 2-sigma unlucky initial error made the gate reject every measurement;
   the filter never recovered (352 rejections, 3 accepted updates) and the robot drove into the wall. Fix: lock-out recovery (inflate P).
2. *Collision at an obstacle (tuning seed 8).* Estimate was good (3.9 cm RMSE); the likely cause (not isolated by an experiment) was replanning every 2 s making the
   path jink around a 14 cm radius obstacle with 19 cm inflation. Change made: replan only on a blocked path, inflation 0.24 m.

## 7. Honest limitations
* **Simulator**: NumPy stand-in with the same sensor interface a MuJoCo backend needs (`src/sim/mujoco_backend.py` is an unimplemented stub).
  Contact dynamics, wheel-ground physics, rendering and a real ArUco detector (cv2.aruco) are not exercised.
* The camera is simulated at the measurement level (pixel column and size with Gaussian noise). Real detections add outliers, motion blur,
  latency and pose-estimation bias; a 5% calibration error is already visible in NEES.
* Heading is only observable from the camera through motion or two markers; stretches with no marker in view (up to several seconds at the
  start, before the camera sees a wall) are bridged by odometry alone.
* Obstacles are cylinders only; the LiDAR planner has no dynamic obstacles. Start region and waypoint are fixed, so the trials are not all
  possible rooms. Tracking RMSE includes the start-up transient.
* Metrics are on 30 to 50 trials: a 100% success rate over 30 trials is consistent with a true rate down to about 89%.

## 8. What is still yours to do
- [ ] Implement the MuJoCo backend (`src/sim/mujoco_backend.py`) with the Week 11 TurtleBot3 model and re-run `evaluate.py`. Expect to re-tune q_scale / r_scale.
- [ ] Replace the simulated detector with real `cv2.aruco` detection from the MuJoCo camera and run calibration on rendered views.
- [ ] Record the 5-minute video (`python make_demo.py` gives a starting GIF: true vs estimate, 2-sigma ellipse, error vs 2-sigma).
- [ ] Write the contribution statement and the viva preparation (below). Each of you must be able to explain every module.
- [ ] Optional HD material: UKF vs EKF, active obstacle avoidance, hardware.

## 9. Contribution statement
*(to be completed by the team: who did what)*

## 10. Generative AI declaration
*(to be completed by the team. Draft: "Claude (Anthropic) was used to generate the initial structure and code of this
repository, adapted from the Week 5 to 9 tutorial notebooks, and to draft this README. The team reviewed, ran and modified ...")*
Declaring this is required by the marking guide.

## 11. Viva preparation (the marking guide's prompts, with where the answers are)
* Why does it look like that? Open `results/run_seed3.png`: the dead-reckoning error grows steadily, the EKF error stays inside 2 sigma, shrinking
  as the robot nears markers (R grows with range squared).
* Observable states / marker leaves view: section 2 analysis; one marker alone gives rank 2 when stationary.
* Where do Q and R come from, what if R were 10x smaller: sections 3 and 4; R x0.1 row (NEES 10.4, overconfident).
* How were the LQR weights chosen, what limits them: section 5; R sweep.
* Frames from measurement to controller: pixels -> (range, bearing) in C -> offset 0.06 m -> pose in W (EKF) -> error in B (tracker) -> (v, w).
* What failed: section 6 failures.
