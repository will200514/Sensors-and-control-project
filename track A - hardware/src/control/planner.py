"""Global planning on the LiDAR occupancy grid (A*) and trajectory generation.
turns the LiDAR occupancy map into a route, then adds the headings and speeds needed by the LQR controller."""
import heapq
import numpy as np
from scipy.ndimage import distance_transform_edt
from .. import config as C

# adds obstacle clearence
# it allows the model to treat the robot as a point, while creating greater clearence around the objects
# to allows room for the objects body to clear
def inflated_map(grid, inflate):
    occ = grid.occupied()
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True          # finite map extent only, not simulator room geometry
    dist = distance_transform_edt(~occ) * grid.res
    return dist < inflate


# finds a route 
# A serches for a low cost route between the start and the desired destination
def astar(blocked, grid, start_xy, goal_xy):
    s = tuple(int(c) for c in grid.to_cell(*start_xy))
    g = tuple(int(c) for c in grid.to_cell(*goal_xy))
    blocked = blocked.copy()
    for cell in (s, g):                       # never refuse because the start/goal cell is inflated
        r = 4
        i0, i1 = max(cell[0] - r, 0), min(cell[0] + r + 1, grid.n)
        j0, j1 = max(cell[1] - r, 0), min(cell[1] + r + 1, grid.m)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        blocked[i0:i1, j0:j1] &= np.hypot(ii - cell[0], jj - cell[1]) > r
    nbrs = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    openq, came, cost = [(0.0, s)], {}, {s: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return [grid.cell_to_world(i, j) for i, j in path[::-1]]
        for di, dj, c in nbrs:
            n = (cur[0] + di, cur[1] + dj)
            if not (0 <= n[0] < grid.n and 0 <= n[1] < grid.m) or blocked[n]:
                continue
            nc = cost[cur] + c
            if nc < cost.get(n, 1e9):
                cost[n], came[n] = nc, cur
                dx, dy = abs(n[0] - g[0]), abs(n[1] - g[1])
                h = (dx + dy) + (1.414 - 2) * min(dx, dy)
                heapq.heappush(openq, (nc + h, n)); 
    return None



# checks a straight connection
# checks wether a straight section connect two points without entering a blocked cell
def _free_line(blocked, grid, p, q):
    n = max(int(np.linalg.norm(np.array(q) - np.array(p)) / (grid.res / 2)), 2)
    xs, ys = np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)
    ix, iy = grid.to_cell(xs, ys)
    return not blocked[ix, iy].any()



# removes unnesesary intermediate points
def shortcut(path, blocked, grid):
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _free_line(blocked, grid, path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return np.array(out)


# rounds the paths cornrs
def chaikin(P, iters=3):
    for _ in range(iters):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(P[-1]); P = np.array(Q)
    return P

# choses a path smoothing level
def smooth_path(pts, blocked, grid):
    pts = np.array(pts, float)
    for it in (3, 2, 1, 0):
        sm = chaikin(pts, it) 
        ix, iy = grid.to_cell(sm[:, 0], sm[:, 1])
        if not blocked[ix, iy].any():
            return sm
    return pts



# adds heading and reference speeds
# 
def make_trajectory(P, v_ref, v_end, dt=C.DT, a_dec=0.08, v_floor=0.04):
    """Time-parameterise a polyline: trapezoid-style speed profile, w_ref from path curvature."""
    P = np.array(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.hstack([[True], seg > 1e-6])
    P, seg = P[keep], seg[keep[1:]]
    s_cum = np.hstack([[0.0], np.cumsum(seg)])
    S = s_cum[-1]
    heading = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0]))
    heading = np.hstack([heading, heading[-1]])
    heading = np.unwrap(heading)
    X, s = [], 0.0
    while s < S:
        v = max(min(v_ref, np.sqrt(v_end ** 2 + 2 * a_dec * (S - s))), v_floor)
        x = np.interp(s, s_cum, P[:, 0]); y = np.interp(s, s_cum, P[:, 1])
        th = np.interp(s, s_cum, heading)
        X.append([x, y, th, v]); s += v * dt
    X.append([P[-1, 0], P[-1, 1], heading[-1], v_floor])
    X = np.array(X)
    w = np.gradient(X[:, 2]) / dt
    w = np.clip(w, -0.8 * C.W_MAX, 0.8 * C.W_MAX)
    k = 5                                                    # light smoothing of omega_ref
    w = np.convolve(np.pad(w, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return np.column_stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3], w])   # x, y, th, v_r, w_r


