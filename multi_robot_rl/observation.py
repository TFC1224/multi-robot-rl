"""Shared observation construction utilities.

Used by both :class:`MultiAgentExplorerEnv` (training) and
:class:`AgentNode` (deployment) so that observation logic stays in sync.

Typical usage::

    from multi_robot_rl.observation import build_observation, extract_local_map

    # During training (env):
    local_crop = extract_local_map(world.local_occupancy[i], (r, c), ...)
    obs = build_observation(
        local_crop, (r, c, theta), grid_shape,
        teammate_poses=[(rj, cj, thetaj) for ...],
        shared_occupancy=world.shared_occupancy,
        n_agents=N_AGENTS, max_frontiers=MAX_FRONTIERS,
    )

    # During deployment (ROS node):
    shared = _to_numpy(shared_map_msg)
    local_crop = extract_local_map(shared, (r, c), ...)
    obs = build_observation(
        local_crop, (r, c, theta), shared.shape,
        teammate_poses=resolved_teammate_poses,
        shared_occupancy=shared,
        n_agents=N_AGENTS, max_frontiers=MAX_FRONTIERS,
    )
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .frontier_detector import detect_frontiers

# Occupancy grid pixel values (match grid_world constants).
FREE = 0
OCCUPIED = 1
UNKNOWN = 2


def extract_local_map(
    occupancy: np.ndarray,
    center: Tuple[int, int],
    local_map_size: int = 64,
    *,
    free_val: int = FREE,
    occupied_val: int = OCCUPIED,
    unknown_val: int = UNKNOWN,
) -> np.ndarray:
    """Extract a 3-channel local map crop centered on *center*.

    Args:
        occupancy: (H, W) uint8 occupancy grid.
        center: (row, col) of the agent.
        local_map_size: output square size (default 64).

    Returns:
        (3, local_map_size, local_map_size) float32 array:
        ``[free_ch, obstacle_ch, explored_ch]``.
    """
    S = local_map_size
    H, W = occupancy.shape
    r, c = center
    half = S // 2
    r0, r1 = r - half, r - half + S
    c0, c1 = c - half, c - half + S

    src_r0 = max(r0, 0)
    src_r1 = min(r1, H)
    src_c0 = max(c0, 0)
    src_c1 = min(c1, W)

    out = np.full((S, S), unknown_val, dtype=np.uint8)
    if src_r1 > src_r0 and src_c1 > src_c0:
        dst_r0 = src_r0 - r0
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = src_c0 - c0
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        out[dst_r0:dst_r1, dst_c0:dst_c1] = occupancy[src_r0:src_r1, src_c0:src_c1]

    ch_free = (out == free_val).astype(np.float32)
    ch_obstacle = (out == occupied_val).astype(np.float32)
    ch_explored = ((out == free_val) | (out == occupied_val)).astype(np.float32)
    return np.stack([ch_free, ch_obstacle, ch_explored], axis=0)


def _normalize_pose(r: float, c: float, theta: float,
                    height: int, width: int) -> np.ndarray:
    """Return the 4-d normalized own-pose vector."""
    return np.array([
        r / max(height - 1, 1) * 2.0 - 1.0,
        c / max(width - 1, 1) * 2.0 - 1.0,
        float(np.sin(theta)),
        float(np.cos(theta)),
    ], dtype=np.float32)


def _teammate_feature(r: float, c: float, theta: float,
                      tr: float, tc: float, t_theta: float,
                      height: int, width: int) -> np.ndarray:
    """Return the 4-d relative teammate feature vector."""
    return np.array([
        (tr - r) / max(height - 1, 1),
        (tc - c) / max(width - 1, 1),
        float(np.sin(t_theta - theta)),
        float(np.cos(t_theta - theta)),
    ], dtype=np.float32)


def build_observation(
    local_map: np.ndarray,
    own_pose_rc_theta: Tuple[int, int, float],
    grid_shape: Tuple[int, int],
    *,
    teammate_poses: Optional[List[Optional[Tuple[int, int, float]]]] = None,
    shared_occupancy: Optional[np.ndarray] = None,
    n_agents: int = 3,
    max_frontiers: int = 16,
) -> dict:
    """Build a single-agent observation dict from raw inputs.

    This is the canonical observation builder shared by the training
    environment and the ROS deployment node.

    Args:
        local_map: (3, S, S) local map crop (already extracted).
        own_pose_rc_theta: (row, col, yaw) of this agent.
        grid_shape: (height, width) of the full grid.
        teammate_poses: list of (row, col, yaw) for each other agent.
            May contain ``None`` entries for missing data (ROS case).
        shared_occupancy: (H, W) shared occupancy grid (for frontier detection).
        n_agents: total number of agents.
        max_frontiers: max frontier candidates.

    Returns:
        Dict with keys ``local_map``, ``own_pose``, ``teammates``,
        ``frontiers``, ``n_frontiers``.
    """
    H, W = grid_shape
    r, c, theta = own_pose_rc_theta

    own_pose = _normalize_pose(r, c, theta, H, W)

    teammates = np.zeros((n_agents - 1, 4), dtype=np.float32)
    if teammate_poses:
        slot = 0
        for tp in teammate_poses:
            if slot >= n_agents - 1:
                break
            if tp is not None:
                tr, tc, t_theta = tp
                teammates[slot] = _teammate_feature(
                    r, c, theta, tr, tc, t_theta, H, W)
            slot += 1

    # Frontiers
    if shared_occupancy is not None:
        frontiers, n_valid = detect_frontiers(
            shared_occupancy, (r, c), max_n=max_frontiers)
    else:
        frontiers = np.zeros((max_frontiers, 3), dtype=np.float32)
        n_valid = 0

    return {
        'local_map': local_map,
        'own_pose': own_pose,
        'teammates': teammates,
        'frontiers': frontiers,
        'n_frontiers': int(n_valid),
    }
