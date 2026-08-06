"""Multi-agent Gymnasium environment wrapping the 2D grid world.

Implements the standard Gymnasium API plus the multi-agent returns expected
by MAPPO (tuple observations, tuple rewards, joint done). Designed so the
observation tensors match the DistributedActor input spec in networks.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .frontier_detector import detect_frontiers
from .grid_world import FREE, OCCUPIED, UNKNOWN, GridWorld
from .observation import build_observation, extract_local_map
from .reward_functions import RewardWeights, compute_team_reward, jain_index


# Local map window size around each robot (must match networks.DistributedActor).
LOCAL_MAP_SIZE = 64
MAX_FRONTIERS = 16
N_AGENTS = 3


@dataclass
class EnvConfig:
    scenario: str = 'multi_1'
    max_steps_per_episode: int = 300
    target_coverage: float = 0.95
    step_cells: int = 5
    reward_weights: RewardWeights | None = None


class MultiAgentExplorerEnv(gym.Env):
    """Pure-CTDE 2D multi-robot exploration environment."""

    metadata = {'render_modes': ['rgb_array']}
    render_fps = 5

    def __init__(self, config: dict | EnvConfig | None = None,
                 scenario: str | None = None):
        super().__init__()

        if isinstance(config, EnvConfig):
            self.env_cfg = config
        else:
            cfg = dict(config or {})
            if scenario is not None:
                cfg['scenario'] = scenario
            self.env_cfg = EnvConfig(
                scenario=cfg.get('scenario', 'multi_1'),
                max_steps_per_episode=cfg.get('max_steps_per_episode', 300),
                target_coverage=cfg.get('target_coverage', 0.95),
                step_cells=cfg.get('step_cells', 5),
                reward_weights=cfg.get('reward_weights'),
            )

        self.world = GridWorld(self.env_cfg.scenario)
        self.n_agents = N_AGENTS
        self.world.n_agents = N_AGENTS

        # Joint action space: each agent picks a frontier index (0..MAX_FRONTIERS-1)
        self.action_space = spaces.MultiDiscrete([MAX_FRONTIERS] * N_AGENTS)

        # Per-agent observation space
        self.observation_space = spaces.Tuple([
            spaces.Dict({
                'local_map': spaces.Box(
                    low=0.0, high=1.0,
                    shape=(3, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.float32),
                'own_pose': spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                'teammates': spaces.Box(
                    low=-1.0, high=1.0, shape=(N_AGENTS - 1, 4), dtype=np.float32),
                'frontiers': spaces.Box(
                    low=-10.0, high=10.0,
                    shape=(MAX_FRONTIERS, 3), dtype=np.float32),
                'n_frontiers': spaces.Discrete(MAX_FRONTIERS + 1),
            })
            for _ in range(N_AGENTS)
        ])

        self._step_cells = self.env_cfg.step_cells
        self._target_coverage = self.env_cfg.target_coverage
        self._max_steps = self.env_cfg.max_steps_per_episode

        self._weights = self.env_cfg.reward_weights or RewardWeights()
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.step_count = 0
        self.team_coverage = 0.0
        self.individual_coverage = np.zeros(self.n_agents, dtype=np.float32)
        self.collision_count = 0
        self.world.reset()
        # First-step sensing so we observe the spawn neighborhood
        for i in range(self.n_agents):
            self.world.sense(i)
        self.world.merge()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[tuple[dict, ...], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.world.seed = seed
        self._reset_state()
        return self._get_observations(), {}

    def step(self, actions: np.ndarray | list[int]) -> tuple[
        tuple[dict, ...], tuple[float, ...], bool, bool, dict[str, Any]
    ]:
        actions = np.asarray(actions).flatten()
        assert len(actions) == self.n_agents, (
            f'Expected {self.n_agents} actions, got {len(actions)}')

        prev_team = self.team_coverage
        prev_individual = self.individual_coverage.copy()

        # 1. Pick targets from current frontier list (re-detect after merge)
        self.world.merge()
        targets: list[tuple[int, int]] = []
        for i in range(self.n_agents):
            shared = self.world.shared_occupancy
            r, c, _ = self.world.poses[i]
            fronts, n_valid = detect_frontiers(
                shared, (r, c), max_n=MAX_FRONTIERS)
            action = int(actions[i])
            if n_valid == 0 or action >= n_valid:
                targets.append((r, c))  # stay put
            else:
                # De-normalize frontier coordinates
                rr = int(round(fronts[action, 0] * (shared.shape[0] - 1)))
                cc = int(round(fronts[action, 1] * (shared.shape[1] - 1)))
                rr = int(np.clip(rr, 0, shared.shape[0] - 1))
                cc = int(np.clip(cc, 0, shared.shape[1] - 1))
                targets.append((rr, cc))

        # 2. Move all agents (in parallel — sleep step_time would be ros-time)
        for i in range(self.n_agents):
            self.world.move_toward(i, targets[i], step_cells=self._step_cells)

        # 3. Sensing (each agent sees new cells around its new pose)
        for i in range(self.n_agents):
            self.world.sense(i)
        self.world.merge()

        # 4. Collision detection
        collision = self.world.check_collision()
        if collision:
            self.collision_count += 1

        # 5. Update coverage metrics
        self.team_coverage = self.world.team_coverage()
        self.individual_coverage = np.array(
            self.world.individual_coverage(), dtype=np.float32)
        overlap = self.world.overlap_ratio()

        # 6. Reward
        done = self.team_coverage >= self._target_coverage
        team_r, individual_r = compute_team_reward(
            prev_team_coverage=prev_team,
            curr_team_coverage=self.team_coverage,
            prev_individual=prev_individual,
            curr_individual=self.individual_coverage,
            overlap_ratio=overlap,
            collision=collision,
            done=done,
            weights=self._weights,
        )
        # Subtract collision-induced damage from individual reward (already in
        # team reward via team_share); apply locally too so policies learn.
        if collision:
            individual_r -= 5.0

        self.step_count += 1
        truncated = self.step_count >= self._max_steps
        terminated = bool(done)

        observations = self._get_observations()
        info = {
            'team_coverage': self.team_coverage,
            'individual_coverage': self.individual_coverage.tolist(),
            'jain_index': jain_index(self.individual_coverage),
            'overlap_ratio': overlap,
            'collision_count': self.collision_count,
            'team_reward': team_r,
        }
        return (
            tuple(observations),
            tuple(float(x) for x in individual_r),
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_observations(self) -> tuple[dict, ...]:
        return tuple(self._build_agent_obs(i) for i in range(self.n_agents))

    def _build_agent_obs(self, agent_id: int) -> dict[str, np.ndarray]:
        r, c, theta = self.world.poses[agent_id]
        H, W = self.world.config.height, self.world.config.width

        local_map = extract_local_map(
            self.world.local_occupancy[agent_id], (r, c),
            local_map_size=LOCAL_MAP_SIZE,
            free_val=FREE, occupied_val=OCCUPIED, unknown_val=UNKNOWN)

        teammate_poses = [
            self.world.poses[j] for j in range(self.n_agents) if j != agent_id
        ]

        return build_observation(
            local_map, (r, c, theta), (H, W),
            teammate_poses=teammate_poses,
            shared_occupancy=self.world.shared_occupancy,
            n_agents=self.n_agents,
            max_frontiers=MAX_FRONTIERS,
        )

    # ------------------------------------------------------------------
    # Rendering (optional)
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | None:
        if self.render_mode != 'rgb_array':
            return None
        H, W = self.world.config.height, self.world.config.width
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        img[self.world.shared_occupancy == OCCUPIED] = (40, 40, 40)
        img[self.world.shared_occupancy == UNKNOWN] = (200, 200, 200)
        # Mark explored cells per agent with a faint tint
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        for i in range(self.n_agents):
            explored = self.world.explored[i] & (
                self.world.static_occupancy == FREE)
            img[explored] = (
                (img[explored].astype(np.int32) + np.array(colors[i])) // 2
            ).astype(np.uint8)
            r, c, _ = self.world.poses[i]
            img[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = colors[i]
        return img

    # ------------------------------------------------------------------
    # Helpers used by the trainer
    # ------------------------------------------------------------------

    def build_global_state(self) -> dict[str, np.ndarray]:
        """Build the centralized critic input. Includes:
            - shared_map       (3, H', W') centered crop for the critic
            - robot_positions  (n_agents*2,)
            - robot_oris       (n_agents*2,) sin/cos
            - team_coverage    (1,)
            - step_count       (1,)
        """
        H, W = self.world.config.height, self.world.config.width
        # Center-crop shared map to a fixed 64x64 to match actor input.
        S = LOCAL_MAP_SIZE
        cr = self.world.config.height // 2
        cc = self.world.config.width // 2
        half = S // 2
        r0, r1 = max(0, cr - half), min(H, cr - half + S)
        c0, c1 = max(0, cc - half), min(W, cc - half + S)
        crop = np.full((S, S), UNKNOWN, dtype=np.uint8)
        dst_r0 = r0 - (cr - half)
        dst_r1 = dst_r0 + (r1 - r0)
        dst_c0 = c0 - (cc - half)
        dst_c1 = dst_c0 + (c1 - c0)
        crop[dst_r0:dst_r1, dst_c0:dst_c1] = self.world.shared_occupancy[r0:r1, c0:c1]
        ch_free = (crop == FREE).astype(np.float32)
        ch_obstacle = (crop == OCCUPIED).astype(np.float32)
        ch_explored = ((crop == FREE) | (crop == OCCUPIED)).astype(np.float32)
        shared = np.stack([ch_free, ch_obstacle, ch_explored], axis=0)

        positions = np.zeros(self.n_agents * 2, dtype=np.float32)
        oris = np.zeros(self.n_agents * 2, dtype=np.float32)
        for i, (r, c, th) in enumerate(self.world.poses):
            positions[2 * i] = r / max(H - 1, 1) * 2 - 1
            positions[2 * i + 1] = c / max(W - 1, 1) * 2 - 1
            oris[2 * i] = float(np.sin(th))
            oris[2 * i + 1] = float(np.cos(th))
        return {
            'shared_map': shared,
            'robot_positions': positions,
            'robot_oris': oris,
            'team_coverage': np.array([self.team_coverage], dtype=np.float32),
            'step_count': np.array(
                [self.step_count / max(self._max_steps, 1)], dtype=np.float32),
        }
