"""Training entry point for the multi_robot_rl ROS 2 package.

Supports both the legacy CLI mode (every hyperparameter as a flag) and
the GRALP-inspired JSON config mode (``--config config/train_config.json``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from multi_robot_rl.config_loader import load_all_configs, merge_cli_overrides
from multi_robot_rl.mappo_trainer import MAPPOTrainer, TrainConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Train MAPPO for multi-robot collaborative exploration.')

    # ── config mode ──────────────────────────────────────────────
    p.add_argument('--config', type=str, default='config/train_config.json',
                   help='Path to train_config.json (default: config/train_config.json)')

    # ── scenario & environment ───────────────────────────────────
    p.add_argument('--scenario', type=str, default=None,
                   choices=['multi_1', 'multi_2', 'multi_3'])
    p.add_argument('--n_agents', type=int, default=None)
    p.add_argument('--target_coverage', type=float, default=None)
    p.add_argument('--max_steps_per_episode', type=int, default=None)

    # ── PPO hyperparameters ──────────────────────────────────────
    p.add_argument('--total_timesteps', type=int, default=None)
    p.add_argument('--n_steps', type=int, default=None)
    p.add_argument('--n_epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--gae_lambda', type=float, default=None)
    p.add_argument('--clip_range', type=float, default=None)
    p.add_argument('--actor_lr', type=float, default=None)
    p.add_argument('--critic_lr', type=float, default=None)
    p.add_argument('--entropy_coef', type=float, default=None)
    p.add_argument('--value_loss_coef', type=float, default=None)
    p.add_argument('--max_grad_norm', type=float, default=None)
    p.add_argument('--amp', action='store_true', default=None)

    # ── logging & checkpointing ──────────────────────────────────
    p.add_argument('--save_freq', type=int, default=None)
    p.add_argument('--log_freq', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--device', type=str, default=None)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 1. Load config from JSON
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to cwd
        config_path = Path.cwd() / config_path
    configs = load_all_configs(config_path)

    # 2. Merge CLI overrides (only non-None values)
    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k != 'config'}
    if cli_overrides:
        configs = merge_cli_overrides(configs, cli_overrides)

    # 3. Build TrainConfig
    train_cfg = TrainConfig.from_configs(configs)
    train = configs['train']

    # 4. Log the effective configuration
    print(f"[train] config : {configs['_paths']['train']}")
    print(f"[train] env    : {configs['_paths']['env']}")
    print(f"[train] model  : {configs['_paths']['model']}")
    print(f"[train] device : {train_cfg.device}")
    print(f"[train] scenario: {train_cfg.scenario}")
    print(f"[train] total steps: {train_cfg.total_timesteps:,}")
    print(f"[train] AMP    : {train_cfg.use_amp} ({train_cfg.amp_dtype})")

    # 5. Train
    trainer = MAPPOTrainer(train_cfg)
    trainer.train()
    return 0


if __name__ == '__main__':
    sys.exit(main())
