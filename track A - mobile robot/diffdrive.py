"""State-space model of the TurtleBot3 (differential drive) and the LQR error model.

State x = [x, y, theta]; odometry input u = [v, omega] with
    v = (v_R + v_L) / 2,   omega = (v_R - v_L) / b .
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from ..config import TRACK


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def wheels_to_body(vl, vr, b=TRACK):
    return 0.5 * (vr + vl), (vr - vl) / b


def body_to_wheels(v, w, b=TRACK):
    return v - 0.5 * w * b, v + 0.5 * w * b


def _sinc_half(w, dt):
    """sin(a)/a with a = w dt / 2, and its derivative with respect to w."""
    a = 0.5 * w * dt
    if abs(a) < 1e-4:
        s = 1.0 - a * a / 6.0
        ds_da = -a / 3.0
    else:
        s = np.sin(a) / a
        ds_da = (a * np.cos(a) - np.sin(a)) / (a * a)
    return s, ds_da * 0.5 * dt


def f_motion(x, u, dt):
    """Exact constant-(v, omega) arc over dt (same model as the Week 7/8 tutorials, written
    in a form that has no singularity at omega = 0)."""
    v, w = u
    s, _ = _sinc_half(w, dt)
    thm = x[2] + 0.5 * w * dt
    return np.array([x[0] + v * dt * s * np.cos(thm),
                     x[1] + v * dt * s * np.sin(thm),
                     float(wrap(x[2] + w * dt))])


def F_motion(x, u, dt):
    """df/dx."""
    v, w = u
    s, _ = _sinc_half(w, dt)
    thm = x[2] + 0.5 * w * dt
    F = np.eye(3)
    F[0, 2] = -v * dt * s * np.sin(thm)
    F[1, 2] = v * dt * s * np.cos(thm)
    return F


def G_motion(x, u, dt):
    """df/du, used to propagate odometry noise into process noise."""
    v, w = u
    s, ds_dw = _sinc_half(w, dt)
    thm = x[2] + 0.5 * w * dt
    G = np.zeros((3, 2))
    G[0, 0] = dt * s * np.cos(thm)
    G[1, 0] = dt * s * np.sin(thm)
    G[0, 1] = v * dt * (ds_dw * np.cos(thm) - s * 0.5 * dt * np.sin(thm))
    G[1, 1] = v * dt * (ds_dw * np.sin(thm) + s * 0.5 * dt * np.cos(thm))
    G[2, 1] = dt
    return G


# ---- tracking-error model for the LQR -------------------------------------------------------
def error_model(v_r, w_r):
    """Continuous error dynamics in the robot frame (Kanayama), linearised about e = 0.

    e = [e_x, e_y, e_theta] = [R(theta)^T (p_ref - p), theta_ref - theta]; u = u_ref + du.
        de/dt = A e + B du
    """
    A = np.array([[0.0, w_r, 0.0],
                  [-w_r, 0.0, v_r],
                  [0.0, 0.0, 0.0]])
    B = np.array([[-1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, -1.0]])
    return A, B


def c2d(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])


def obsv_window(Fs, Hs):
    """Stacked observability matrix for a time-varying pair: rows H_k F_{k-1} ... F_0."""
    rows, Phi = [], np.eye(Fs[0].shape[0])
    for F, H in zip(Fs, Hs):
        Phi = F @ Phi
        rows.append(H @ Phi)
    return np.vstack(rows)
