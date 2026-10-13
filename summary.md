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
