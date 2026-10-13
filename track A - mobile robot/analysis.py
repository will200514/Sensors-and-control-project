"""Part A: controllability of the tracking-error model and observability of the EKF linearisation."""
import numpy as np
from src import config as C
from src.model.diffdrive import error_model, ctrb, F_motion, obsv_window
from src.perception.camera import H_cam

np.set_printoptions(precision=4, suppress=True)
out = []
out.append("CONTROLLABILITY of the error model (A(v_r, w_r), B)  [rank must be 3]")
out.append(f"{'v_r':>6}{'w_r':>6}{'rank':>6}{'sigma_min(ctrb)':>18}")
for v, w in [(0.18, 0.0), (0.18, 0.5), (0.04, 0.0), (0.01, 0.0), (0.0, 0.5), (0.0, 0.0)]:
    M = ctrb(*error_model(v, w))
    out.append(f"{v:6.2f}{w:6.2f}{np.linalg.matrix_rank(M):6d}{np.linalg.svd(M, compute_uv=False).min():18.4f}")
out.append("-> at v_r = 0 and w_r = 0 the lateral error e_y cannot be corrected (rank 2): a robot must move to fix a sideways offset.")
out.append("   With v_r = 0, w_r != 0 the model is controllable, but only because the reference itself is rotating.\n")

out.append("OBSERVABILITY of the EKF linearisation over a 1 s window (20 steps), pose (1, 1, 0)")
x = np.array([1.0, 1.0, 0.0])
def window(v, w, markers):
    Fs = [F_motion(x, (v, w), C.DT)] * 20
    Hs = [np.vstack([H_cam(x, m) for m in markers]) if markers else np.zeros((1, 3))] * 20
    O = obsv_window(Fs, Hs)
    return np.linalg.matrix_rank(O, tol=1e-6), np.linalg.svd(O, compute_uv=False)
cases = [("no marker visible, moving", 0.15, 0.0, []),
         ("1 marker ahead, stationary", 0.0, 0.0, [np.array([3.0, 1.0])]),
         ("1 marker ahead, moving", 0.15, 0.0, [np.array([3.0, 1.0])]),
         ("1 marker off-axis, moving", 0.15, 0.0, [np.array([3.0, 3.0])]),
         ("2 markers, stationary", 0.0, 0.0, [np.array([3.0, 1.0]), np.array([1.0, 3.0])]),
         ("2 markers, moving", 0.15, 0.0, [np.array([3.0, 1.0]), np.array([1.0, 3.0])])]
out.append(f"{'case':<32}{'rank':>5}  singular values")
for name, v, w, mk in cases:
    r, s = window(v, w, mk)
    out.append(f"{name:<32}{r:>5}  {s}")
out.append("-> one marker gives range+bearing (2 equations) for 3 states: not instantaneously observable. Heading is recovered\n"
           "   only through motion (position-heading coupling in F) or a second marker. With no marker, only the predicted\n"
           "   covariance exists: P grows without bound (see dead-reckoning error in results/).")
open("results/analysis.txt", "w").write("\n".join(out)); print("\n".join(out))
