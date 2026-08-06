"""Console entry point: python3 scripts/train_mappo.py ...

This script is the canonical training entry point. It mirrors the
hyperparameter set defined in config/mappo_config.yaml.
"""

from __future__ import annotations

import argparse
import sys

from multi_robot_rl.mappo_trainer import MAPPOTrainer, TrainConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Train MAPPO for multi-robot collaborative exploration.')
    p.add_argument('--scenario', type=str, default='multi_1',
                   choices=['multi_1', 'multi_2', 'multi_3'])
    p.add_argument('--total_timesteps', type=int, default=200_000)
    p.add_argument('--n_steps', type=int, default=2048)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--clip_range', type=float, default=0.2)
    p.add_argument('--actor_lr', type=float, default=3e-4)
    p.add_argument('--critic_lr', type=float, default=5e-4)
    p.add_argument('--entropy_coef', type=float, default=0.01)
    p.add_argument('--value_loss_coef', type=float, default=0.5)
    p.add_argument('--save_freq', type=int, default=20_000)
    p.add_argument('--log_freq', type=int, default=2_048)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_dir', type=str, default='models')
    p.add_argument('--target_coverage', type=float, default=0.95)
    p.add_argument('--max_steps_per_episode', type=int, default=300)
    p.add_argument('--device', type=str, default='auto')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = TrainConfig(
        scenario=args.scenario,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        seed=args.seed,
        save_dir=args.save_dir,
        target_coverage=args.target_coverage,
        max_steps_per_episode=args.max_steps_per_episode,
        device=args.device,
    )
    trainer = MAPPOTrainer(cfg)
    trainer.train()
    return 0


if __name__ == '__main__':
    sys.exit(main())
