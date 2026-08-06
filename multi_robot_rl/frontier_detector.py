"""Frontier detection on a known occupancy grid.

A frontier cell is a FREE cell that borders an UNKNOWN cell. We cluster
neighboring frontier cells into candidate goals via simple flood-fill and
return up to ``max_n`` centroids sorted by distance to the robot.

The detection is fully vectorized with NumPy and matches the algorithm in
方案二 §A.1 with a NumPy-only implementation (no OpenCV dependency).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .grid_world import FREE, UNKNOWN


def detect_frontiers(
    shared_occupancy: np.ndarray,
    robot_rc: Tuple[int, int],
    max_n: int = 16,
    cluster_radius: int = 3,
) -> Tuple[np.ndarray, int]:
    """Return ``(frontiers, n_valid)``.

    Parameters
    ----------
    shared_occupancy : (H, W) uint8
        The shared merged map (0=free, 1=occupied, 2=unknown).
    robot_rc : (row, col)
        Current robot position used to sort candidates by distance.
    max_n : int
        Maximum number of returned frontier centroids.
    cluster_radius : int
        Minimum spacing (in cells) between frontier centroids.

    Returns
    -------
    frontiers : (max_n, 3) float32 array
        Columns are ``(row_norm, col_norm, dist_norm)`` matching the
        actor input spec. Padding rows are zeroed.
    n_valid : int
        Number of valid frontier entries (<= max_n).
    """
    free_mask = (shared_occupancy == FREE).astype(np.uint8)
    unknown_mask = (shared_occupancy == UNKNOWN).astype(np.uint8)

    # A frontier cell = FREE cell with at least one UNKNOWN 8-neighbor.
    # Compute via 3x3 convolution (kernel sums 8 neighbors).
    H, W = shared_occupancy.shape
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neigh_unknown = _conv2d_same(unknown_mask.astype(np.int32), kernel)
    frontier_mask = (free_mask.astype(bool) & (neigh_unknown > 0))

    if not frontier_mask.any():
        return np.zeros((max_n, 3), dtype=np.float32), 0

    rs, cs = np.where(frontier_mask)
    centroids = _cluster_centroids(rs, cs, cluster_radius)
    if centroids.size == 0:
        return np.zeros((max_n, 3), dtype=np.float32), 0

    # Sort by distance to robot (closest first)
    dr = centroids[:, 0].astype(np.float32) - robot_rc[0]
    dc = centroids[:, 1].astype(np.float32) - robot_rc[1]
    dist = np.sqrt(dr * dr + dc * dc)
    order = np.argsort(dist)
    centroids = centroids[order]
    dist = dist[order]

    n_valid = min(len(centroids), max_n)
    out = np.zeros((max_n, 3), dtype=np.float32)
    out[:n_valid, 0] = centroids[:n_valid, 0] / max(H - 1, 1)
    out[:n_valid, 1] = centroids[:n_valid, 1] / max(W - 1, 1)
    out[:n_valid, 2] = dist[:n_valid] / max(float(np.hypot(H, W)), 1.0)
    return out, n_valid


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _conv2d_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-mode convolution with zero padding (NumPy implementation).

    Used so we don't pull in scipy as a hard dependency. For our 3x3
    kernel this is fast enough.
    """
    H, W = x.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.zeros((H + 2 * ph, W + 2 * pw), dtype=x.dtype)
    padded[ph:ph + H, pw:pw + W] = x
    out = np.zeros_like(x)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] == 0:
                continue
            out += kernel[i, j] * padded[i:i + H, j:j + W]
    return out


def _cluster_centroids(rs: np.ndarray, cs: np.ndarray,
                       min_sep: int) -> np.ndarray:
    """Greedy farthest-point clustering: pick the first cell, then pick
    cells that are at least ``min_sep`` cells away from all picked cells,
    until exhausted."""
    if rs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pts = np.stack([rs, cs], axis=1)
    picked: list[int] = [0]
    while True:
        last = pts[picked[-1]]
        d = np.max(np.abs(pts - last), axis=1)
        # Mask out already picked
        mask = np.ones(len(pts), dtype=bool)
        mask[picked] = False
        d_masked = np.where(mask, d, -1)
        idx = int(np.argmax(d_masked))
        if d_masked[idx] < min_sep:
            break
        picked.append(idx)
    return pts[picked]
