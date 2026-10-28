"""Occupancy mapping from MuJoCo LiDAR ranges and the estimated robot pose.
creates a grid ecording which areas are free and which contain obstacles or walls"""
import numpy as np
from .. import config as C


# defines the scan direction
BEAM_ANGLES = np.linspace(-np.pi, np.pi, C.LIDAR_BEAMS, endpoint=False)

# stores the occupancy map
# accumulates the evidences over successive scans
class OccGrid:
    """Log-odds occupancy grid, 5 cm cells over the room, updated with ranges + estimated pose."""
    def __init__(self, res=0.05, l_occ=0.9, l_free=-0.4, clamp=5.0, bounds=C.LOCAL_MAP_BOUNDS):        # reates empty map
        self.res = res          # defines cells size in the grid, smaller cells mean more accurate but more processing power
        self.bounds = bounds
        xmin, xmax, ymin, ymax = bounds
        self.xmin, self.ymin = xmin, ymin
        self.n = int(round((xmax - xmin) / res))
        self.m = int(round((ymax - ymin) / res))
        self.L = np.zeros((self.n, self.m))
        self.l_occ, self.l_free, self.clamp = l_occ, l_free, clamp
        self.steps = np.arange(0.0, C.LIDAR_MAX, res * 0.5)

    # coverts coordinates into array indices
    def to_cell(self, x, y):
        return (np.clip(np.floor((np.asarray(x) - self.xmin) / self.res).astype(int), 0, self.n - 1),
                np.clip(np.floor((np.asarray(y) - self.ymin) / self.res).astype(int), 0, self.m - 1))


    # coverts a cell back into coordinates
    def cell_to_world(self, i, j):
        return self.xmin + (i + 0.5) * self.res, self.ymin + (j + 0.5) * self.res



    # incorperates a lidar scan
    # main mapping function 
    def update(self, pose_est, ranges, beam_angles=None):
        """Update the map using ranges and their robot-relative angles."""
        px, py, heading = pose_est

        ranges = np.asarray(ranges, dtype=float)

        # Preserve compatibility with the simulated LiDAR.
        if beam_angles is None:
            beam_angles = BEAM_ANGLES

        beam_angles = np.asarray(beam_angles, dtype=float)

        if ranges.ndim != 1 or ranges.shape != beam_angles.shape:
            raise ValueError(
                "LiDAR ranges and angles must be matching 1D arrays."
            )

        # Conservatively ignore invalid readings.
        valid = (
            np.isfinite(ranges)
            & np.isfinite(beam_angles)
            & (ranges > 0.0)
            & (ranges <= C.LIDAR_MAX)
        )

        ranges = ranges[valid]
        beam_angles = beam_angles[valid]

        if ranges.size == 0:
            return

        angles = heading + beam_angles
        dx = np.cos(angles)
        dy = np.sin(angles)

        # Clear cells before each measured endpoint.
        distances = self.steps[None, :]
        free = distances < (ranges[:, None] - self.res)

        sample_x = px + distances * dx[:, None]
        sample_y = py + distances * dy[:, None]

        cells = self.to_cell(
            sample_x[free],
            sample_y[free],
        )
        np.add.at(self.L, cells, self.l_free)

        # Mark endpoints below the configured maximum range.
        hit = ranges < C.LIDAR_MAX - 0.02

        hit_x = px + ranges[hit] * dx[hit]
        hit_y = py + ranges[hit] * dy[hit]

        cells = self.to_cell(hit_x, hit_y)
        np.add.at(self.L, cells, self.l_occ)

        np.clip(
            self.L,
            -self.clamp,
            self.clamp,
            out=self.L,
        )


    # supplies the obstacle maap to the planner
    def occupied(self):
        return self.L > 0.5

