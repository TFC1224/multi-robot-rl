"""MAPPO training loop.

Implements a from-scratch MAPPO trainer using the Gymnasium environment
in :mod:`multi_agent_env`. Only the standard PyTorch + NumPy stack is
required (no Stable-Baselines3, no Ray) so it runs identically on the
RTX 3080 training server and on the Jetson Orin NX.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .multi_agent_env import (MAX_FRONTIERS, N_AGENTS, EnvConfig,
                              MultiAgentExplorerEnv)
from .networks import CentralizedCritic, DistributedActor
from .replay_buffer import BufferConfig, RolloutBuffer


@dataclass
class TrainConfig:
    scenario: str = 'multi_1'
    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    save_freq: int = 50_000
    log_freq: int = 2_000
    seed: int = 0
    save_dir: str = 'models'
    target_coverage: float = 0.95
    max_steps_per_episode: int = 300
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'

    # Phase 4 additions
    use_amp: bool = False
    amp_dtype: str = 'bfloat16'
    lr_schedule: str = 'none'  # 'none' | 'linear'

    @classmethod
    def from_configs(cls, configs: Dict[str, Any]) -> 'TrainConfig':
        """Build a TrainConfig from the merged config dict."""
        train = configs.get('train', {})
        env = configs.get('env', {})
        model = configs.get('model', {})
        return cls(
            scenario=env.get('scenario', train.get('scenario', 'multi_1')),
            total_timesteps=train.get('total_timesteps', 1_000_000),
            n_steps=train.get('n_steps', 2048),
            n_epochs=train.get('n_epochs', 10),
            batch_size=train.get('batch_size', 64),
            gamma=train.get('gamma', 0.99),
            gae_lambda=train.get('gae_lambda', 0.95),
            clip_range=train.get('clip_range', 0.2),
            actor_lr=train.get('actor_lr', 3e-4),
            critic_lr=train.get('critic_lr', 5e-4),
            entropy_coef=train.get('entropy_coef', 0.01),
            value_loss_coef=train.get('value_loss_coef', 0.5),
            max_grad_norm=train.get('max_grad_norm', 0.5),
            save_freq=train.get('save_freq', 50_000),
            log_freq=train.get('log_freq', 2_048),
            seed=train.get('seed', 0),
            save_dir=train.get('save_dir', 'models'),
            target_coverage=env.get('target_coverage', train.get('target_coverage', 0.95)),
            max_steps_per_episode=env.get('max_steps_per_episode', train.get('max_steps_per_episode', 300)),
            device=train.get('device', 'auto'),
            use_amp=train.get('amp', False),
            amp_dtype=train.get('amp_dtype', 'bfloat16'),
            lr_schedule=train.get('lr_schedule', 'none'),
        )


class MAPPOTrainer:
    """MAPPO trainer with shared-parameter actor + centralized critic."""

    def __init__(self, config: TrainConfig | None = None):
        self.cfg = config or TrainConfig()
        if self.cfg.device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.cfg.device)

        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.env = MultiAgentExplorerEnv(EnvConfig(
            scenario=self.cfg.scenario,
            target_coverage=self.cfg.target_coverage,
            max_steps_per_episode=self.cfg.max_steps_per_episode,
        ))
        self.n_agents = self.env.n_agents

        self.actor = DistributedActor(n_actions=MAX_FRONTIERS,
                                      n_teammates=self.n_agents - 1).to(self.device)
        self.critic = CentralizedCritic(n_agents=self.n_agents).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=self.cfg.critic_lr)

        self.buffer_cfg = BufferConfig(
            n_steps=self.cfg.n_steps,
            n_agents=self.n_agents,
            max_frontiers=MAX_FRONTIERS,
        )
        self.buffer = RolloutBuffer(self.buffer_cfg)

        Path(self.cfg.save_dir).mkdir(parents=True, exist_ok=True)

        self.episode_returns: list[float] = []
        self.episode_coverages: list[float] = []
        self._current_return = 0.0
        self._current_coverage = 0.0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _obs_batch(self, observations) -> tuple[np.ndarray, ...]:
        local_maps = np.stack([o['local_map'] for o in observations], axis=0)
        own_poses = np.stack([o['own_pose'] for o in observations], axis=0)
        teammates = np.stack([o['teammates'] for o in observations], axis=0)
        frontiers = np.stack([o['frontiers'] for o in observations], axis=0)
        n_frontiers = np.array([o['n_frontiers'] for o in observations],
                               dtype=np.int64)
        return local_maps, own_poses, teammates, frontiers, n_frontiers

    def _global_state_batch(self) -> dict[str, np.ndarray]:
        g = self.env.build_global_state()
        # Add batch dimension
        return {k: v[None, ...] for k, v in g.items()}

    @torch.no_grad()
    def _select_actions(self, observations):
        local_maps, own_poses, teammates, frontiers, n_frontiers = \
            self._obs_batch(observations)
        local_maps_t = torch.from_numpy(local_maps).to(self.device)
        own_poses_t = torch.from_numpy(own_poses).to(self.device)
        teammates_t = torch.from_numpy(teammates).to(self.device)
        frontiers_t = torch.from_numpy(frontiers).to(self.device)
        n_frontiers_t = torch.from_numpy(n_frontiers).to(self.device)

        logits = self.actor(local_maps_t, own_poses_t, teammates_t,
                            frontiers_t, n_frontiers_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        g = self._global_state_batch()
        value = self.critic(
            torch.from_numpy(g['shared_map']).to(self.device),
            torch.from_numpy(g['robot_positions']).to(self.device),
            torch.from_numpy(g['robot_oris']).to(self.device),
            torch.cat([
                torch.from_numpy(g['team_coverage']).to(self.device),
                torch.from_numpy(g['step_count']).to(self.device),
            ], dim=1),
        ).squeeze().item()

        return (actions.cpu().numpy(), log_probs.cpu().numpy(),
                local_maps, own_poses, teammates, frontiers, n_frontiers,
                g['shared_map'][0], g['robot_positions'][0],
                g['robot_oris'][0], np.array([g['team_coverage'][0, 0],
                                              g['step_count'][0, 0]],
                                             dtype=np.float32), value)

    def collect_rollouts(self, n_steps: int) -> None:
        observations, _ = self.env.reset()
        for _ in range(n_steps):
            (actions, log_probs, local_maps, own_poses, teammates,
             frontiers, n_frontiers, shared_map, robot_positions,
             robot_oris, team_stats, value) = self._select_actions(
                observations)

            next_obs, rewards, terminated, truncated, info = self.env.step(
                actions)
            done = bool(terminated or truncated)
            self._current_return += float(sum(rewards))
            self._current_coverage = float(info['team_coverage'])

            self.buffer.add(
                local_maps=local_maps,
                own_poses=own_poses,
                teammates=teammates,
                frontiers=frontiers,
                n_frontiers=n_frontiers,
                shared_map=shared_map,
                robot_positions=robot_positions,
                robot_oris=robot_oris,
                team_stats=team_stats,
                actions=actions,
                log_probs=log_probs,
                value=value,
                rewards=np.asarray(rewards, dtype=np.float32),
                team_reward=float(info.get('team_reward',
                                           float(np.mean(rewards)))),
                done=done,
            )

            if done:
                self.episode_returns.append(self._current_return)
                self.episode_coverages.append(self._current_coverage)
                self._current_return = 0.0
                observations, _ = self.env.reset()
            else:
                observations = next_obs

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        tensors = self.buffer.to_tensors(self.device)

        # Bootstrap value for the state after the last collected step.
        # We re-run critic on the current env observation for simplicity.
        with torch.no_grad():
            g = self._global_state_batch()
            last_value = self.critic(
                torch.from_numpy(g['shared_map']).to(self.device),
                torch.from_numpy(g['robot_positions']).to(self.device),
                torch.from_numpy(g['robot_oris']).to(self.device),
                torch.cat([
                    torch.from_numpy(g['team_coverage']).to(self.device),
                    torch.from_numpy(g['step_count']).to(self.device),
                ], dim=1),
            ).squeeze().item()
            last_done = bool(self.env.step_count >= self.env._max_steps)
        adv, ret = self.buffer.compute_advantages(last_value, last_done)
        advantages = torch.from_numpy(adv).to(self.device)
        returns = torch.from_numpy(ret).to(self.device)

        n = self.buffer.cfg.n_steps
        a = self.buffer.cfg.n_agents

        # Flatten (T, A, ...) -> (T*A, ...) for minibatch sampling
        local_maps = tensors['local_maps'].reshape(n * a, 3, 64, 64)
        own_poses = tensors['own_poses'].reshape(n * a, 4)
        teammates = tensors['teammates'].reshape(n * a, a - 1, 4)
        frontiers = tensors['frontiers'].reshape(n * a, MAX_FRONTIERS, 3)
        n_frontiers = tensors['n_frontiers'].reshape(n * a)
        actions = tensors['actions'].reshape(n * a)
        old_lp = tensors['old_log_probs'].reshape(n * a)

        adv_flat = advantages.unsqueeze(1).expand(n, a).reshape(n * a)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        critic_losses = []
        actor_losses = []
        entropies = []

        idxs = np.arange(n)
        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.batch_size):
                mb = idxs[start:start + self.cfg.batch_size]
                if len(mb) == 0:
                    continue
                actor_loss_epoch = 0.0
                entropy_epoch = 0.0
                for agent_i in range(a):
                    mb_full = mb * a + agent_i
                    mb_t = torch.from_numpy(mb_full).long().to(self.device)

                    mb_local = local_maps.index_select(0, mb_t)
                    mb_pose = own_poses.index_select(0, mb_t)
                    mb_team = teammates.index_select(0, mb_t)
                    mb_front = frontiers.index_select(0, mb_t)
                    mb_nf = n_frontiers.index_select(0, mb_t)
                    mb_act = actions.index_select(0, mb_t)
                    mb_lp = old_lp.index_select(0, mb_t)
                    mb_adv = adv_flat.index_select(0, mb_t)

                    logits = self.actor(mb_local, mb_pose, mb_team,
                                        mb_front, mb_nf)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - mb_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_range,
                        1.0 + self.cfg.clip_range,
                    ) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss_epoch += actor_loss.item()
                    entropy_epoch += entropy.item()

                    loss = actor_loss - self.cfg.entropy_coef * entropy
                    self.actor_optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.cfg.max_grad_norm)
                    self.actor_optim.step()

                # ---- critic update (one minibatch per step indices, since
                # critic input is global-state only) ----
                mb_idx = torch.from_numpy(mb).long().to(self.device)
                v_pred = self.critic(
                    tensors['shared_maps'].reshape(n, 3, 64, 64)
                        .index_select(0, mb_idx),
                    tensors['robot_positions'].reshape(n, a * 2)
                        .index_select(0, mb_idx),
                    tensors['robot_oris'].reshape(n, a * 2)
                        .index_select(0, mb_idx),
                    tensors['team_stats'].reshape(n, 2)
                        .index_select(0, mb_idx),
                ).squeeze(-1)
                v_target = returns[mb_idx]
                critic_loss = F.mse_loss(v_pred, v_target)
                self.critic_optim.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optim.step()

                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss_epoch / a)
                entropies.append(entropy_epoch / a)

        return {
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
            'entropy': float(np.mean(entropies)) if entropies else 0.0,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int | None = None) -> None:
        total = total_timesteps or self.cfg.total_timesteps
        print(f'[MAPPO] Starting training: {total} steps on {self.device}')
        start = time.time()
        for timestep in range(0, total, self.cfg.n_steps):
            self.collect_rollouts(self.cfg.n_steps)
            metrics = self.update()

            if (timestep // self.cfg.n_steps) % max(
                    1, self.cfg.log_freq // self.cfg.n_steps) == 0:
                elapsed = time.time() - start
                sps = (timestep + self.cfg.n_steps) / max(elapsed, 1e-3)
                avg_return = (np.mean(self.episode_returns[-10:])
                              if self.episode_returns else 0.0)
                avg_cov = (np.mean(self.episode_coverages[-10:])
                           if self.episode_coverages else 0.0)
                print(
                    f'[MAPPO] step={timestep + self.cfg.n_steps:>8d} '
                    f'sps={sps:6.1f} '
                    f'return={avg_return:7.2f} '
                    f'coverage={avg_cov:.3f} '
                    f'actor_loss={metrics["actor_loss"]:+.4f} '
                    f'critic_loss={metrics["critic_loss"]:+.4f} '
                    f'entropy={metrics["entropy"]:.4f}'
                )

            if ((timestep + self.cfg.n_steps) % self.cfg.save_freq == 0
                    and (timestep + self.cfg.n_steps) > 0):
                self.save(timestep + self.cfg.n_steps)

        self.save(total)

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def save(self, timestep: int) -> None:
        torch.save(self.actor.state_dict(),
                   os.path.join(self.cfg.save_dir, f'mappo_actor_{timestep}.pth'))
        torch.save(self.critic.state_dict(),
                   os.path.join(self.cfg.save_dir, f'mappo_critic_{timestep}.pth'))
        torch.save(self.actor.state_dict(),
                   os.path.join(self.cfg.save_dir, 'mappo_actor.pth'))
        torch.save(self.critic.state_dict(),
                   os.path.join(self.cfg.save_dir, 'mappo_critic.pth'))
        print(f'[MAPPO] Saved checkpoint at step {timestep}')

    def load(self, path_actor: str, path_critic: str | None = None) -> None:
        self.actor.load_state_dict(torch.load(path_actor, map_location=self.device))
        if path_critic:
            self.critic.load_state_dict(
                torch.load(path_critic, map_location=self.device))
        self.actor.eval()
        self.critic.eval()

    def config_dict(self) -> dict:
        return asdict(self.cfg)
