"""On-policy rollout buffer for MAPPO.

Stores per-step tensors for every agent and exposes the GAE computation
needed for the PPO update. We store *shared* tensors for fields shared by
all agents (shared map) and per-agent tensors for everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BufferConfig:
    n_steps: int = 2048
    n_agents: int = 3
    max_frontiers: int = 16
    map_size: int = 64
    n_teammates: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95


class RolloutBuffer:
    """Per-step storage for one rollout (n_steps transitions)."""

    def __init__(self, cfg: BufferConfig):
        self.cfg = cfg
        n = cfg.n_steps
        a = cfg.n_agents

        # Per-agent fields
        self.local_maps = np.zeros(
            (n, a, 3, cfg.map_size, cfg.map_size), dtype=np.float32)
        self.own_poses = np.zeros((n, a, 4), dtype=np.float32)
        self.teammates = np.zeros(
            (n, a, cfg.n_teammates, 4), dtype=np.float32)
        self.frontiers = np.zeros(
            (n, a, cfg.max_frontiers, 3), dtype=np.float32)
        self.n_frontiers = np.zeros((n, a), dtype=np.int64)
        self.actions = np.zeros((n, a), dtype=np.int64)
        self.log_probs = np.zeros((n, a), dtype=np.float32)
        self.rewards = np.zeros((n, a), dtype=np.float32)
        self.team_rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=np.float32)

        # Global fields (shared map + team info)
        self.shared_maps = np.zeros((n, 3, cfg.map_size, cfg.map_size),
                                    dtype=np.float32)
        self.robot_positions = np.zeros((n, a * 2), dtype=np.float32)
        self.robot_oris = np.zeros((n, a * 2), dtype=np.float32)
        self.team_stats = np.zeros((n, 2), dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)

        self._idx = 0

    def add(self, *, local_maps, own_poses, teammates, frontiers,
            n_frontiers, shared_map, robot_positions, robot_oris,
            team_stats, actions, log_probs, value, rewards, team_reward,
            done) -> None:
        i = self._idx
        self.local_maps[i] = local_maps
        self.own_poses[i] = own_poses
        self.teammates[i] = teammates
        self.frontiers[i] = frontiers
        self.n_frontiers[i] = n_frontiers
        self.shared_maps[i] = shared_map
        self.robot_positions[i] = robot_positions
        self.robot_oris[i] = robot_oris
        self.team_stats[i] = team_stats
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = value
        self.rewards[i] = rewards
        self.team_rewards[i] = team_reward
        self.dones[i] = float(done)
        self._idx += 1

    def __len__(self) -> int:
        return self._idx

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_advantages(self, last_value: float,
                           last_done: bool) -> tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns based on team rewards + values."""
        n = len(self)
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        next_non_terminal = 0.0 if last_done else 1.0
        for t in reversed(range(n)):
            if self.dones[t]:
                next_non_terminal = 0.0
                next_value = 0.0
                last_gae = 0.0
            delta = self.team_rewards[t] + gamma * next_value * next_non_terminal \
                - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
            next_non_terminal = 1.0 - self.dones[t]
        returns = advantages + self.values[:n]
        return advantages, returns

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            'local_maps': torch.from_numpy(self.local_maps).to(device),
            'own_poses': torch.from_numpy(self.own_poses).to(device),
            'teammates': torch.from_numpy(self.teammates).to(device),
            'frontiers': torch.from_numpy(self.frontiers).to(device),
            'n_frontiers': torch.from_numpy(self.n_frontiers).to(device),
            'shared_maps': torch.from_numpy(self.shared_maps).to(device),
            'robot_positions': torch.from_numpy(self.robot_positions).to(device),
            'robot_oris': torch.from_numpy(self.robot_oris).to(device),
            'team_stats': torch.from_numpy(self.team_stats).to(device),
            'actions': torch.from_numpy(self.actions).to(device),
            'old_log_probs': torch.from_numpy(self.log_probs).to(device),
            'values': torch.from_numpy(self.values).to(device),
            'rewards': torch.from_numpy(self.rewards).to(device),
            'team_rewards': torch.from_numpy(self.team_rewards).to(device),
            'dones': torch.from_numpy(self.dones).to(device),
        }
