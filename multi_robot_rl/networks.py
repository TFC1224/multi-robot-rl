"""Neural network architectures for MAPPO.

- DistributedActor: per-agent policy network operating on local observations
  (local map crop, own pose, teammate poses, frontier candidates).
  Parameters are shared across all three agents.
- CentralizedCritic: receives the full shared map + all agent poses +
  team-level statistics to produce a single value estimate. Only used
  during training; not deployed on the robots.

Mirrors 方案三 §4.3-4.4 (CentralizedCritic / DistributedActor). Uses only
PyTorch primitives so the package can be built without torchvision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ortho_init(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal weight initialization for Conv2d and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


# ----------------------------------------------------------------------
# Reusable building blocks
# ----------------------------------------------------------------------

class MapEncoder(nn.Module):
    """3-layer CNN that compresses a 3xSxS map crop to a 128-d vector.

    Identical to the encoder used in 方案二's QNetwork so we can swap
    weights and compare behaviours across single- and multi-agent setups.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 128,
                 ortho_init: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # S/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # S/4
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # S/8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, out_dim)
        if ortho_init:
            self.apply(_ortho_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(self.conv(x)))


# ----------------------------------------------------------------------
# Distributed actor (parameter-shared across agents)
# ----------------------------------------------------------------------

class DistributedActor(nn.Module):
    """Per-agent policy network. Returns action logits over MAX_FRONTIERS."""

    def __init__(self, n_actions: int = 16, map_channels: int = 3,
                 n_teammates: int = 2, ortho_init: bool = False,
                 ortho_gain: float = 1.0):
        super().__init__()
        self.n_actions = n_actions
        self.n_teammates = n_teammates

        self.map_encoder = MapEncoder(in_channels=map_channels, out_dim=128,
                                      ortho_init=ortho_init)
        self.pose_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.teammate_encoder = nn.Sequential(
            nn.Linear(n_teammates * 4, 64), nn.ReLU())
        self.frontier_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 32 + 64 + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        if ortho_init:
            self.apply(_ortho_init)
            # Policy final layer → small init for exploration (GRALP convention)
            last_linear = self.policy_head[-1]
            nn.init.orthogonal_(last_linear.weight, gain=ortho_gain)
            nn.init.constant_(last_linear.bias, 0)

    def forward(self, local_map: torch.Tensor, own_pose: torch.Tensor,
                teammates: torch.Tensor, frontiers: torch.Tensor,
                n_frontiers: torch.Tensor) -> torch.Tensor:
        """Return (B, n_actions) logits."""
        B = local_map.size(0)
        map_feat = self.map_encoder(local_map)                   # [B, 128]
        pose_feat = self.pose_encoder(own_pose)                  # [B, 32]
        team_feat = self.teammate_encoder(teammates.flatten(1))  # [B, 64]

        # Frontiers: [B, max_n, 3] -> encode each then masked-mean pool
        frontier_feat = self.frontier_encoder(frontiers)         # [B, max_n, 128]
        idx = torch.arange(frontiers.size(1), device=frontiers.device)
        idx = idx.unsqueeze(0).expand(B, -1)
        mask = (idx < n_frontiers.unsqueeze(1)).float().unsqueeze(-1)
        frontier_feat = (frontier_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        combined = torch.cat([map_feat, pose_feat, team_feat, frontier_feat], dim=1)
        return self.policy_head(combined)                        # [B, n_actions]


# ----------------------------------------------------------------------
# Centralized critic (only used during training)
# ----------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """Centralized value network. Sees the shared map + all agent poses."""

    def __init__(self, n_agents: int = 3, map_channels: int = 3,
                 ortho_init: bool = False):
        super().__init__()
        self.n_agents = n_agents

        self.map_encoder = MapEncoder(in_channels=map_channels, out_dim=128,
                                      ortho_init=ortho_init)
        self.pos_encoder = nn.Sequential(
            nn.Linear(n_agents * 2, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.team_encoder = nn.Sequential(
            nn.Linear(n_agents * 2 + 2, 32), nn.ReLU(),  # oris + (coverage, step)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

        if ortho_init:
            self.apply(_ortho_init)

    def forward(self, shared_map: torch.Tensor,
                robot_positions: torch.Tensor, robot_oris: torch.Tensor,
                team_stats: torch.Tensor) -> torch.Tensor:
        """Return (B, 1) value estimate.

        ``team_stats`` is a (B, 2) tensor with ``[team_coverage, step_norm]``.
        """
        map_feat = self.map_encoder(shared_map)                 # [B, 128]
        pos_feat = self.pos_encoder(robot_positions)            # [B, 64]
        team_feat = self.team_encoder(
            torch.cat([robot_oris, team_stats], dim=1))          # [B, 32]
        combined = torch.cat([map_feat, pos_feat, team_feat], dim=1)
        return self.value_head(combined)                        # [B, 1]
