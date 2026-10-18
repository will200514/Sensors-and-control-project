"""Occupancy mapping from MuJoCo LiDAR ranges and the estimated robot pose."""
import numpy as np
from .. import config as C

BEAM_ANGLES = np.linspace(-np.pi, np.pi, C.LIDAR_BEAMS, endpoint=False)


class OccGrid:
    """Log-odds occupancy grid, 5 cm cells over the room, updated with ranges + estimated pose."""
    def __init__(self, res=0.05, l_occ=0.9, l_free=-0.4, clamp=5.0):
        self.res = res
        xmin, xmax, ymin, ymax = C.ROOM
        self.n = int(round((xmax - xmin) / res))
        self.m = int(round((ymax - ymin) / res))
        self.L = np.zeros((self.n, self.m))
        self.l_occ, self.l_free, self.clamp = l_occ, l_free, clamp
        self.steps = np.arange(0.0, C.LIDAR_MAX, res * 0.5)

    def to_cell(self, x, y):
        return (np.clip((np.asarray(x) / self.res).astype(int), 0, self.n - 1),
                np.clip((np.asarray(y) / self.res).astype(int), 0, self.m - 1))

    def update(self, pose_est, ranges):
        px, py, th = pose_est
        a = th + BEAM_ANGLES
        t = self.steps[None, :]
        r = ranges[:, None]
        free = t < (r - self.res)
        X = px + t * np.cos(a)[:, None]
        Y = py + t * np.sin(a)[:, None]
        ix, iy = self.to_cell(X[free], Y[free])
        np.add.at(self.L, (ix, iy), self.l_free)
        hit = ranges < C.LIDAR_MAX - 0.02
        hx, hy = self.to_cell(px + ranges[hit] * np.cos(a[hit]), py + ranges[hit] * np.sin(a[hit]))
        np.add.at(self.L, (hx, hy), self.l_occ)
        np.clip(self.L, -self.clamp, self.clamp, out=self.L)

    def occupied(self):
        return self.L > 0.5
