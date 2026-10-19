"""Extended Kalman filter: wheel odometry (prediction) + camera range/bearing to ArUco markers."""
import numpy as np
from .. import config as C
from ..model.diffdrive import f_motion, F_motion, G_motion, wrap, body_to_wheels
from ..perception.camera import h_cam, H_cam, R_cam, pixels_to_rb

# estimates the uncertainty in the wheel dometry
# answers how much error should we expect in the speeds used to predict movement
def odom_covariance(v, w, q_scale=1.0):
    """Covariance of the odometry velocity input, derived from wheel-level noise.

    Wheel surface speeds vL, vR; encoder noise e (sigma_enc) and relative slip eps (sigma_slip, plus
    a per-trial bias sigma_bias that we cannot estimate, so it is folded into the slip term):
        v_err   ~ (e_L + e_R)/2          - (vL eps_L + vR eps_R)/2
        w_err   ~ (e_R - e_L)/b          - (vR eps_R - vL eps_L)/b
    """
    vl, vr = body_to_wheels(v, w)              # converts body speed into left and right wheel speeds
    s2 = C.SIGMA_SLIP ** 2 + C.SIGMA_BIAS ** 2  # two wheel source errors
    var_v = C.SIGMA_ENC ** 2 / 2 + s2 * (vl ** 2 + vr ** 2) / 4
    var_w = 2 * C.SIGMA_ENC ** 2 / C.TRACK ** 2 + s2 * (vl ** 2 + vr ** 2) / C.TRACK ** 2
    cov_vw = s2 * (vr ** 2 - vl ** 2) / (2 * C.TRACK) * (-1)   # shared slip terms correlate v and w
    return q_scale * np.array([[var_v, cov_vw], [cov_vw, var_w]])


# creates and initiallises the filter
# runs when a file creates an EKF
# x0 = Initial estimated position and heading
# P0 = Initial uncertainty in that estimate
# cfg = Filter settings, including noise scales, focal length, and rejection threshold
class EKF:
    def __init__(self, x0, P0, cfg):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)
        self.cfg = cfg
        self.n_updates = 0
        self.n_rejected = 0
        self.last_nis = []
        self.frame_rej = 0
        self.n_recoveries = 0


    # predicts movement nd increases uncertainty
    # For example, the wheels might indicate that the robot travelled one metre. 
    # The estimate moves forward by approximately that amount, while the covariance accounts for possible encoder errors and slip.
    def predict(self, u, dt=C.DT):
        F = F_motion(self.x, u, dt)
        G = G_motion(self.x, u, dt)
        Qd = G @ odom_covariance(*u, q_scale=self.cfg.q_scale) @ G.T + np.diag([1e-8, 1e-8, 1e-8])
        self.x = f_motion(self.x, u, dt)
        self.P = F @ self.P @ F.T + Qd

    # corrects the estimate using one marker
    # This is the EKF’s measurement-update step. It processes one ArUco marker observation.
    # estimates camera uncertainty
    # uses the innovation covariance to combine:
    #Uncertainty in the predicted observation caused by pose uncertainty.
    #Uncertainty in the camera measurement.
    # computes NIS: NIS measures how surprising the disagreement is relative to the expected uncertainty.
    def update_marker(self, mk_xy, u_px, s_px):
        cfg = self.cfg
        rho_m, beta_m = pixels_to_rb(u_px, s_px, fx=cfg.fx_filter)
        z = np.array([rho_m, beta_m])
        R = cfg.r_scale * R_cam(rho_m, fx=cfg.fx_filter)
        zhat = h_cam(self.x, mk_xy)
        H = H_cam(self.x, mk_xy)
        nu = z - zhat
        nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        if cfg.gate is not None and nis > cfg.gate:
            self.n_rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form keeps P symmetric PSD
        self.n_updates += 1
        self.last_nis.append(nis)
        return True


    # handles repeated rejection of camera observations
    # This function helps the filter recover when its estimate becomes so inaccurate that the measurement gate rejects every observation.
    # makes filter less confident so it accepts measuremetents when 5 have been rejected
    def end_frame(self, n_detected, n_accepted):
        """Lock-out recovery. A gate protects against outliers, but a filter that is genuinely
        wrong rejects every measurement and never recovers. If every detection in 5 consecutive
        camera frames (0.5 s) is rejected, the filter admits it may be lost: inflate P so the
        gate re-opens and the next detections can pull the estimate back."""
        if n_detected == 0:
            return
        self.frame_rej = 0 if n_accepted > 0 else self.frame_rej + 1
        if self.frame_rej >= 5:
            self.P = self.P + np.diag([0.3 ** 2, 0.3 ** 2, np.deg2rad(20.0) ** 2])
            self.frame_rej = 0
            self.n_recoveries += 1
