"""2D grid world scenarios for Multi-1 / Multi-2 / Multi-3.

Encodes the exploration environment as an occupancy grid:
    0 = free (currently explored as free)
    1 = obstacle (walls, boxes)
    2 = unknown

The grid mimics the Gazebo worlds from 训练实验文档-方案三 §2.2-2.4
but is fully NumPy-based so training can run without Gazebo / ROS 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


FREE = 0
OCCUPIED = 1
UNKNOWN = 2


@dataclass
class GridWorldConfig:
    """Static structural configuration of a scenario."""

    name: str
    height: int          # rows (y axis)
    width: int           # cols (x axis)
    resolution: float    # meters per cell
    free_area: int       # number of FREE cells (excluding obstacles)
    spawn_positions: list[tuple[int, int]] = field(default_factory=list)


# ----------------------------------------------------------------------
# Scenario builders
# ----------------------------------------------------------------------

def _multi_1() -> tuple[np.ndarray, GridWorldConfig]:
    """Multi-1: 5m x 7m (50x70 @ 0.1m) double room, 4 boxes."""
    H, W = 50, 70
    grid = np.full((H, W), FREE, dtype=np.uint8)

    # Outer walls (thickness 1)
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED

    # Vertical divider with doorway (split into 2 rooms)
    grid[10:40, 35] = OCCUPIED
    grid[5:9, 35] = OCCUPIED
    grid[41:46, 35] = OCCUPIED  # doorway gap rows 9-10 and 40-41

    # 4 paper boxes scattered in the rooms
    boxes = [(12, 12), (12, 58), (38, 12), (38, 58)]
    for r, c in boxes:
        _stamp_box(grid, r, c, 4)

    # 3 spawn positions spread across the rooms + corridor
    spawn = [(8, 8), (8, 60), (25, 35)]

    free_area = int((grid == FREE).sum())
    return grid, GridWorldConfig('multi_1', H, W, 0.1, free_area, spawn)


def _multi_2() -> tuple[np.ndarray, GridWorldConfig]:
    """Multi-2: L-shape with concave corners, ~50 m^2, 6 boxes."""
    H, W = 60, 70
    grid = np.full((H, W), FREE, dtype=np.uint8)

    # Outer L-shape walls
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED

    # Carve out the L-shape by blocking the top-right quadrant
    # Outer rectangle of free area: rows 5..54, cols 5..64
    # Internal wall making the L: rows 5..30 in cols 35..38
    grid[5:30, 35] = OCCUPIED
    grid[5:30, 38] = OCCUPIED

    # Concave wall poking into the bottom-right free area
    grid[40:44, 50:55] = OCCUPIED

    # 6 boxes
    boxes = [(15, 12), (15, 50), (45, 12), (45, 50),
             (25, 20), (50, 30)]
    for r, c in boxes:
        _stamp_box(grid, r, c, 4)

    spawn = [(10, 10), (10, 55), (45, 55)]

    free_area = int((grid == FREE).sum())
    return grid, GridWorldConfig('multi_2', H, W, 0.1, free_area, spawn)


def _multi_3() -> tuple[np.ndarray, GridWorldConfig]:
    """Multi-3: 8m x 10m (80x100 @ 0.1m) multi-room + long corridor, 8 boxes."""
    H, W = 80, 100
    grid = np.full((H, W), FREE, dtype=np.uint8)

    # Outer walls
    grid[0, :] = OCCUPIED
    grid[-1, :] = OCCUPIED
    grid[:, 0] = OCCUPIED
    grid[:, -1] = OCCUPIED

    # Horizontal wall dividing into top/bottom halves (with corridor gap)
    grid[40, 10:45] = OCCUPIED
    grid[40, 55:95] = OCCUPIED  # doorway cols 45-54

    # Vertical wall splitting bottom into 2 rooms
    grid[40:75, 50] = OCCUPIED
    grid[42:46, 50] = OCCUPIED  # doorway rows 40-41

    # Vertical wall splitting top half
    grid[5:38, 70] = OCCUPIED
    grid[5:9, 70] = OCCUPIED    # doorway rows 9-10

    # Long corridor along the top
    # (already free in the top strip rows 5-10)

    # 8 boxes
    boxes = [(15, 20), (15, 80), (25, 30), (25, 60),
             (60, 20), (60, 80), (70, 30), (70, 70)]
    for r, c in boxes:
        _stamp_box(grid, r, c, 4)

    spawn = [(8, 15), (8, 85), (60, 50)]

    free_area = int((grid == FREE).sum())
    return grid, GridWorldConfig('multi_3', H, W, 0.1, free_area, spawn)


def _stamp_box(grid: np.ndarray, r: int, c: int, size: int) -> None:
    """Stamp a square obstacle box of given size at (r, c) center."""
    H, W = grid.shape
    half = size // 2
    r0, r1 = max(0, r - half), min(H, r - half + size)
    c0, c1 = max(0, c - half), min(W, c - half + size)
    grid[r0:r1, c0:c1] = OCCUPIED


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

_BUILDERS = {
    'multi_1': _multi_1,
    'multi_2': _multi_2,
    'multi_3': _multi_3,
}


class GridWorld:
    """Wrapper holding the structural map, the visibility map, and helpers.

    The visibility map records which cells have been seen by at least one robot.
    The ``static_occupancy`` attribute is fixed and represents the true walls
    and obstacles. ``visible_occupancy`` overlays unknown cells (2) on top.
    """

    def __init__(self, scenario: str = 'multi_1', seed: int = 0):
        if scenario not in _BUILDERS:
            raise ValueError(
                f'Unknown scenario {scenario!r}; choose from {list(_BUILDERS)}')

        static_grid, self.config = _BUILDERS[scenario]()
        self.scenario = scenario
        self.seed = seed
        self.resolution = self.config.resolution  # m / cell

        # True structural map (never changes during an episode)
        self.static_occupancy: np.ndarray = static_grid.copy()

        # Per-robot visibility / explored map (3D: agents x H x W)
        # Channels per agent: 0=free seen, 1=obstacle seen, 2=unknown
        self.n_agents = 3
        self.local_occupancy: np.ndarray = np.full(
            (self.n_agents, self.config.height, self.config.width),
            UNKNOWN, dtype=np.uint8,
        )
        # Shared merged occupancy (the analog of map_merge output)
        self.shared_occupancy: np.ndarray = np.full(
            (self.config.height, self.config.width),
            UNKNOWN, dtype=np.uint8,
        )
        # Per-agent visited history (boolean)
        self.visited: list[np.ndarray] = [
            np.zeros((self.config.height, self.config.width), dtype=bool)
            for _ in range(self.n_agents)
        ]
        # Per-agent explored cells (cumulative free cells ever seen)
        self.explored: list[np.ndarray] = [
            np.zeros((self.config.height, self.config.width), dtype=bool)
            for _ in range(self.n_agents)
        ]
        # Robot poses (row, col, theta)
        self.poses: list[tuple[int, int, float]] = list(
            self.config.spawn_positions[: self.n_agents]
        )

    # ---- reset / state ------------------------------------------------

    def reset(self) -> None:
        """Reset the visibility layer while keeping the static map intact."""
        self.local_occupancy[:] = UNKNOWN
        self.shared_occupancy[:] = UNKNOWN
        for v in self.visited:
            v[:] = False
        for e in self.explored:
            e[:] = False
        self.poses = list(self.config.spawn_positions[: self.n_agents])

    # ---- sensing ------------------------------------------------------

    def sensor_range_cells(self) -> int:
        """Sensor range in cells (4 m for the simulated 2D laser)."""
        return int(4.0 / self.resolution)

    def sense(self, robot_id: int, fov_deg: float = 360.0) -> None:
        """Update the local visibility map for ``robot_id``.

        Uses a circular footprint with the given field of view. With fov_deg
        == 360 the entire circle is revealed (omnidirectional LIDAR).
        """
        r, c, _ = self.poses[robot_id]
        H, W = self.config.height, self.config.width
        R = self.sensor_range_cells()

        rr, cc = np.ogrid[:H, :W]
        mask = (rr - r) ** 2 + (cc - c) ** 2 <= R ** 2
        mask &= self.static_occupancy != OCCUPIED
        # Cells beyond walls are not visible
        mask &= _line_of_sight(self.static_occupancy, r, c, R)

        local = self.local_occupancy[robot_id]
        local[mask] = self.static_occupancy[mask]
        self.visited[robot_id][mask] = True
        self.explored[robot_id][mask & (self.static_occupancy == FREE)] = True

    def merge(self) -> None:
        """Combine all local maps into the shared map (simulated map_merge)."""
        # A cell is "known" if any robot has observed it.
        known = np.any(self.local_occupancy != UNKNOWN, axis=0)
        for v in (FREE, OCCUPIED):
            pick = np.any(self.local_occupancy == v, axis=0)
            self.shared_occupancy[pick] = v
        self.shared_occupancy[~known] = UNKNOWN

    # ---- movement -----------------------------------------------------

    def move_toward(self, robot_id: int, target_rc: tuple[int, int],
                    step_cells: int = 5) -> None:
        """Move the robot toward ``target_rc`` by up to ``step_cells`` cells.

        Stops just before obstacles. Collision is detected via proximity to
        other robots.
        """
        r, c, theta = self.poses[robot_id]
        tr, tc = target_rc
        dr = tr - r
        dc = tc - c
        dist = float(np.hypot(dr, dc))
        if dist < 0.5:
            return
        # Don't overshoot the target: cap step at dist.
        step = max(1, min(step_cells, int(np.floor(dist))))
        nr = int(round(r + step * dr / dist))
        nc = int(round(c + step * dc / dist))
        nr = int(np.clip(nr, 0, self.config.height - 1))
        nc = int(np.clip(nc, 0, self.config.width - 1))

        # Block movement into obstacles: slide along the dominant axis.
        if self.static_occupancy[nr, nc] == OCCUPIED:
            for cand in ((r, nc), (nr, c)):
                cr, cc = cand
                if 0 <= cr < self.config.height and 0 <= cc < self.config.width \
                        and self.static_occupancy[cr, cc] != OCCUPIED:
                    nr, nc = cr, cc
                    break
            else:
                return

        # Update heading
        theta = float(np.arctan2(dr, dc))
        self.poses[robot_id] = (nr, nc, theta)

    def check_collision(self, threshold_cells: int = 2) -> bool:
        """Return True if any two robots are within ``threshold_cells``."""
        for i in range(self.n_agents):
            ri, ci, _ = self.poses[i]
            for j in range(i + 1, self.n_agents):
                rj, cj, _ = self.poses[j]
                if (ri - rj) ** 2 + (ci - cj) ** 2 <= threshold_cells ** 2:
                    return True
        return False

    # ---- metrics ------------------------------------------------------

    def team_coverage(self) -> float:
        """Fraction of free cells that have been explored by any robot."""
        total_free = int((self.static_occupancy == FREE).sum())
        if total_free == 0:
            return 0.0
        any_explored = np.any(
            [e for e in self.explored], axis=0
        ) & (self.static_occupancy == FREE)
        return float(any_explored.sum()) / total_free

    def individual_coverage(self) -> list[float]:
        """Per-robot fraction of free cells explored."""
        total_free = int((self.static_occupancy == FREE).sum())
        if total_free == 0:
            return [0.0] * self.n_agents
        out = []
        for e in self.explored:
            cov = float((e & (self.static_occupancy == FREE)).sum()) / total_free
            out.append(cov)
        return out

    def overlap_ratio(self) -> float:
        """Ratio of cells explored by >= 2 robots, divided by team-explored cells."""
        team = np.any(self.explored, axis=0) & (self.static_occupancy == FREE)
        if not team.any():
            return 0.0
        counts = sum(self.explored).astype(np.int32)
        overlap_cells = int(((counts >= 2) & team).sum())
        return float(overlap_cells) / float(team.sum())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _line_of_sight(static: np.ndarray, r0: int, c0: int,
                   radius: int) -> np.ndarray:
    """Approximate line-of-sight mask.

    A cell within ``radius`` is visible iff every cell along the ray from
    ``(r0, c0)`` to the cell is free of obstacles. We compute this with
    a vectorized Bresenham-style ray sampling.
    """
    H, W = static.shape
    rr, cc = np.ogrid[:H, :W]
    d2 = (rr - r0) ** 2 + (cc - c0) ** 2
    within = d2 <= radius ** 2

    # Build a sorted view of cells by distance from (r0, c0).
    rs, cs = np.where(within)
    if rs.size == 0:
        return within

    dr = rs.astype(np.float32) - r0
    dc = cs.astype(np.float32) - c0
    dist = np.sqrt(dr * dr + dc * dc)
    order = np.argsort(dist)
    rs_o = rs[order]
    cs_o = cs[order]
    dist_o = dist[order]

    # For each ring k=1..max_dist, compute the cell hit at that distance.
    # A cell at distance d is visible iff the maximum occupied distance
    # along its ray is strictly less than d.
    max_step = int(np.ceil(dist_o.max())) + 1
    visible = np.zeros(rs.size, dtype=bool)

    # For each integer step along the ray, sample the cell that lies on the
    # line from the source at that distance; record obstacle hits per cell.
    # Then for each target cell take the minimum hit distance along its ray.
    last_blocked = np.full(rs.size, np.inf, dtype=np.float32)
    for k in range(1, max_step + 1):
        kr = np.round(r0 + dr * k / np.maximum(dist, 1e-3)).astype(np.int32)
        kc = np.round(c0 + dc * k / np.maximum(dist, 1e-3)).astype(np.int32)
        kr = np.clip(kr, 0, H - 1)
        kc = np.clip(kc, 0, W - 1)
        hits = static[kr, kc] == OCCUPIED
        # If hit and k < dist of this cell, the cell is beyond the obstacle
        mask = hits & (k < dist)
        last_blocked = np.where(mask, np.minimum(last_blocked, k).astype(np.float32),
                                last_blocked)

    # A cell is visible iff dist > last_blocked along its ray.
    visible = dist > last_blocked
    out = np.zeros_like(within)
    out[rs[visible], cs[visible]] = True
    return out
