import numpy as np
from src import config as C
from src.model.diffdrive import (f_motion, F_motion, G_motion, error_model, c2d, ctrb, wrap)
from src.perception.camera import h_cam, H_cam


def num_jac(fn, x, eps=1e-6):
    x = np.asarray(x, float); f0 = np.atleast_1d(fn(x)); J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        a, b = np.atleast_1d(fn(x + d)), np.atleast_1d(fn(x - d))
        diff = a - b
        if len(diff) == 2 and fn is not None:
            pass
        J[:, i] = diff / (2 * eps)
    return J


def test_motion_jacobians():
    for u in [(0.15, 0.4), (0.15, 0.0), (0.0, 1.0), (0.1, -0.7)]:
        x = np.array([1.0, 2.0, 0.6])
        Fn = num_jac(lambda s: f_motion(s, u, C.DT), x)
        Gn = num_jac(lambda uu: f_motion(x, uu, C.DT), np.array(u))
        assert np.abs(F_motion(x, u, C.DT) - Fn).max() < 1e-6
        assert np.abs(G_motion(x, u, C.DT) - Gn).max() < 1e-6


def test_camera_jacobian():
    x = np.array([1.0, 2.0, 0.6]); mk = np.array([3.0, 0.5])
    def h(s):
        z = h_cam(s, mk); return z
    Hn = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = 1e-6
        diff = h(x + d) - h(x - d); diff[1] = wrap(diff[1])
        Hn[:, i] = diff / 2e-6
    assert np.abs(H_cam(x, mk) - Hn).max() < 1e-5


def test_controllability_depends_on_speed():
    r = lambda v, w: np.linalg.matrix_rank(ctrb(*error_model(v, w)))
    assert r(0.15, 0.3) == 3
    assert r(0.0, 0.0) == 2          # e_y cannot be corrected without moving