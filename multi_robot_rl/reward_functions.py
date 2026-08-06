"""Team / fairness reward functions for the CTDE MAPPO setup.

Mirrors 方案三 §4.5 — the seven reward components are exposed as toggles
so the ablation studies (-R_team, -R_balance, ...) can be reproduced
without touching the environment code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    """All weight coefficients for the seven reward components."""

    r_explore: float = 10.0
    r_individual: float = 5.0
    r_overlap: float = 3.0
    r_balance: float = 0.5
    r_collision: float = 50.0
    r_step: float = 0.01
    r_done: float = 150.0

    # Ablation switches: set to False to drop the corresponding term.
    use_team: bool = True
    use_individual: bool = True
    use_overlap: bool = True
    use_balance: bool = True
    use_collision: bool = True
    use_step: bool = True
    use_done: bool = True


def jain_index(coverages: np.ndarray) -> float:
    """Jain fairness index. Returns 1.0 when all agents cover the same area."""
    x = np.asarray(coverages, dtype=np.float64)
    if x.size == 0 or x.sum() == 0.0:
        return 1.0
    return float((x.sum() ** 2) / (x.size * (x ** 2).sum()))


def compute_team_reward(
    prev_team_coverage: float,
    curr_team_coverage: float,
    prev_individual: np.ndarray,
    curr_individual: np.ndarray,
    overlap_ratio: float,
    collision: bool,
    done: bool,
    weights: RewardWeights | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the team reward and per-agent decomposition.

    Returns
    -------
    team_reward : float
        Single scalar equal to the team-level reward (used for logging).
    individual_rewards : (n_agents,) ndarray
        Per-agent reward = team_share + individual_share. We use a 0.4/0.6
        split between team and individual contributions as in 方案三 §5.1.
    """
    w = weights or RewardWeights()
    n_agents = len(curr_individual)

    team_r = 0.0
    if w.use_team:
        team_r += w.r_explore * (curr_team_coverage - prev_team_coverage)
    if w.use_overlap:
        team_r -= w.r_overlap * overlap_ratio
    if w.use_balance:
        team_r += w.r_balance * jain_index(curr_individual)
    if w.use_collision and collision:
        team_r -= w.r_collision
    if w.use_step:
        team_r -= w.r_step
    if w.use_done and done:
        team_r += w.r_done

    individual = np.zeros(n_agents, dtype=np.float32)
    if w.use_individual:
        delta_i = curr_individual - prev_individual
        individual += w.r_individual * delta_i

    # Blend team-level and individual contributions 0.4 / 0.6
    team_share = 0.4 * team_r / max(n_agents, 1)
    individual_rewards = team_share + individual
    return float(team_r), individual_rewards
