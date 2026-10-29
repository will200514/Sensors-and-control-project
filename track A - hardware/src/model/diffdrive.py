"""
diffdrive UNICYCLE MODEL
State-space model of the TurtleBot3 (differential drive) and the LQR error model.

contains all of the mathematical models which will be used in the EKF and the LQR control
the motion is modeled off the unicycle kinematic model

State x = [x, y, theta] (robots position and heading); odometry input u = [v, omega] (forward speed and turning) with
    v = (v_R + v_L) / 2 (left and right wheel linear speeds),   omega = (v_R - v_L) / b.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from ..config import TRACK
from .. import config as C


#---------------------------------------------------------------------------
# EKF models
#---------------------------------------------------------------------------

# converts angles into the desired range of -pi<a<pi, since 190 and -170 represents the same angle
# important for heading differences as 190 is the same as -170 but would be calculated as a 360 degree difference
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

# converts the wheel speed into robot movement by calculating velocity v=(vr+vl)/2 and angular velocity w=(vr-vl)/2
# outputs (v,w) used in the motion model
def wheels_to_body(vl, vr, b=TRACK):
    return 0.5 * (vr + vl), (vr - vl) / b

# converts the desired movement into wheel speeds (opposite of wheels to body)
# outoput is (vl,vr) used in the controller to turn required motion into wheel speeds
def body_to_wheels(v, w, b=TRACK):
    return v - 0.5 * w * b, v + 0.5 * w * b

# calculates a stabe curved motion correction
# s is used to erxpress a circular arc motion
# makes turning predictions more accurate in F and G motion
def _sinc_half(w, dt):
    """sin(a)/a with a = w dt / 2, and its derivative with respect to w."""
    a = 0.5 * w * dt
    if abs(a) < 1e-4:             #small angle approximation (ensures a doesnt equal 0 on denominator)
        s = 1.0 - a * a / 6.0     
        ds_da = -a / 3.0
    else:                         #regular circular arc equation for not-small angles
        s = np.sin(a) / a
        ds_da = (a * np.cos(a) - np.sin(a)) / (a * a)
    return s, ds_da * 0.5 * dt    # returns curvred motion correction s, used in G_motion function


# used to predicts the robots next pose
# this is the acutal non linear movement model 
# it extracts speed, calculates the arc correction and finds the heading
# this model predicts x, y, and theta required for the next step in the simulation
# does not model acceleration, wheel slip or collisions
def f_motion(x, u, dt):
    """Exact constant-(v, omega) arc over dt (same model as the Week 7/8 tutorials, written
    in a form that has no singularity at omega = 0)."""
    v, w = u
    s, _ = _sinc_half(w, dt)
    thm = x[2] + 0.5 * w * dt           #curved motion correction
    return np.array([x[0] + v * dt * s * np.cos(thm),     #returns next pose, used in EFK.predict() and model.py
                     x[1] + v * dt * s * np.sin(thm),
                     float(wrap(x[2] + w * dt))])


# defines how uncertainty in the current pose affects the next pose
# calculates the sensitivity to the current pose errors
# includes x and y position errors and heading errors
# ekf uses this to propogate pose covariance
def F_motion(x, u, dt):
    """df/dx."""
    v, w = u
    s, _ = _sinc_half(w, dt)
    thm = x[2] + 0.5 * w * dt
    F = np.eye(3)
    F[0, 2] = -v * dt * s * np.sin(thm) # how heading error affects the predicted positioning
    F[1, 2] = v * dt * s * np.cos(thm)
    return F



# snesitivity to odometry errors
# defines how uncertainty in speed affects the next pose
# calculate the snesitivity to speed inputs
# Together, F and G describe small prediction errors: xnext = Fx + Gu
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
# used in EKF along with F_motion



#---------------------------------------------------------------------------
# ---- tracking-error model for the LQR ------------------------
#---------------------------------------------------------------------------

# builds the linear model used by the LQR controller
# describes error relative to a desired trajectory
# v_r and w_r reference forward and turning speeds
# position errors are expressed in the robot’s coordinate frame:
# e_x = how far behind or ahead the referecne postion is 
# e_y = how far left or right
# e_gheta is the reference heading minus the robots heading
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
    return A, B        # LQRTracker.gain() calls it before discretising the model and calculating a gain.

# converts the contnuous model into a discrete model
# controller runs in steps so it needs a discrete not continuous model for LQR
def c2d(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]

# claculates the optimal feedback gain 
# designs a descrete LQR to balance these two objectives: 
# reduce tracking error
# aovid excessive control corrections
# minimises this cost
# this functions is essentially the heart of the LQR
def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


# ---------------------------------------------------------------------
# controllability and observability
# ---------------------------------------------------------------------

# constructs the controllability matrix
# asks whether the available outputs influence every state direction over time
def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])

# constructs an observability matrix over several updates
def obsv_window(Fs, Hs):
    """Stacked observability matrix for a time-varying pair: rows H_k F_{k-1} ... F_0."""
    rows, Phi = [], np.eye(Fs[0].shape[0])
    for F, H in zip(Fs, Hs):
        Phi = F @ Phi
        rows.append(H @ Phi)
    return np.vstack(rows)




def joint_states_to_body(message):
    """Convert wheel joint velocities from rad/s into robot v and omega."""
    if message is None:
        raise ValueError("No joint-state message received.")

    names = message.get("name", [])
    velocities = message.get("velocity", [])

    if len(names) != len(velocities):
        raise ValueError("Wheel velocity readings are missing.")

    wheel_rates = dict(zip(names, velocities))

    left_rate = wheel_rates["wheel_left_joint"]
    right_rate = wheel_rates["wheel_right_joint"]

    # Wheel angular speed → wheel surface speed in m/s.
    left_speed = left_rate * C.WHEEL_RADIUS
    right_speed = right_rate * C.WHEEL_RADIUS

    return wheels_to_body(left_speed, right_speed)