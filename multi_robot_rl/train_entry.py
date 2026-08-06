"""Training entry point for the multi_robot_rl ROS 2 package.

Supports both the legacy CLI mode (every hyperparameter as a flag) and
the GRALP-inspired JSON config + run-management mode.

Usage::

    # New training run with tag
    python -m multi_robot_rl.train_entry --fresh --tag baseline

    # Resume from a run directory
    python -m multi_robot_rl.train_entry --resume runs/multi_1/20260806-baseline

    # Resume from tagged run
    python -m multi_robot_rl.train_entry --resume baseline

    # Override specific settings
    python -m multi_robot_rl.train_entry --tag experiment --total_timesteps 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from multi_robot_rl.config_loader import load_all_configs, merge_cli_overrides
from multi_robot_rl.mappo_trainer import MAPPOTrainer, TrainConfig
from multi_robot_rl.metric_writer import MetricWriter
from multi_robot_rl.run_manager import RunManager, load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Train MAPPO for multi-robot collaborative exploration.')

    # ── run management (GRALP-style) ─────────────────────────────
    p.add_argument('--fresh', action='store_true', default=True,
                   help='Create a new timestamped run directory (default).')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from a run directory path or tag name.')
    p.add_argument('--tag', type=str, default=None,
                   help='Descriptive tag for the run directory.')
    p.add_argument('--port', type=int, default=None,
                   help='TensorBoard server port (auto-start if set).')
    p.add_argument('--runs-dir', type=str, default='runs',
                   help='Root directory for run storage (default: runs/).')

    # ── config mode ──────────────────────────────────────────────
    p.add_argument('--config', type=str, default='config/train_config.json',
                   help='Path to train_config.json.')

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


def _resolve_resume_path(resume: str, runs_dir: str,
                          scenario: str) -> Path:
    """Resolve a --resume argument to a concrete run directory."""
    candidate = Path(resume)
    if candidate.is_dir():
        return candidate

    # Try as a tag
    rm = RunManager(runs_root=runs_dir, scenario=scenario)
    found = rm.find_run_by_tag(resume)
    if found is not None:
        return found

    raise FileNotFoundError(
        f'Cannot resolve --resume={resume!r}: '
        f'not a directory and no matching tag found in {runs_dir}/{scenario}/')


def _collect_cli_overrides(args) -> Dict[str, Any]:
    """Collect non-None CLI arguments for merging into config."""
    skip = {'fresh', 'resume', 'tag', 'port', 'runs_dir', 'config'}
    return {k: v for k, v in vars(args).items()
            if v is not None and k not in skip}


def main(argv=None):
    args = parse_args(argv)

    # 1. Load config from JSON
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    configs = load_all_configs(config_path)

    # 2. Merge CLI overrides
    cli_overrides = _collect_cli_overrides(args)
    if cli_overrides:
        configs = merge_cli_overrides(configs, cli_overrides)

    train_cfg = TrainConfig.from_configs(configs)
    env_cfg = configs.get('env', {})
    scenario = env_cfg.get('scenario', train_cfg.scenario)

    # 3. Resolve resume path
    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = _resolve_resume_path(
            args.resume, args.runs_dir, scenario)

    # 4. Run directory setup
    run_mgr = RunManager(runs_root=args.runs_dir, scenario=scenario)

    if resume_path is not None:
        run_dir = resume_path
        print(f'[train] resuming from: {run_dir}')
    else:
        run_dir = run_mgr.create_run(tag=args.tag, configs=configs)

    # 5. Create trainer
    if resume_path is not None:
        # Load from checkpoint
        ckpt_path = run_mgr.get_latest_checkpoint(run_dir)
        if ckpt_path is None:
            ckpt_candidates = sorted(
                (run_dir / 'checkpoints').glob('step-*.pt'))
            if ckpt_candidates:
                ckpt_path = ckpt_candidates[-1]
        if ckpt_path is None:
            raise FileNotFoundError(
                f'No checkpoint found in {run_dir}/checkpoints/')
        print(f'[train] loading checkpoint: {ckpt_path}')
        payload = load_checkpoint(ckpt_path, map_location='cpu')
        trainer = MAPPOTrainer(train_cfg)
        trainer.actor.load_state_dict(payload['actor_state_dict'])
        trainer.critic.load_state_dict(payload['critic_state_dict'])
        if payload.get('actor_optim_state_dict'):
            trainer.actor_optim.load_state_dict(payload['actor_optim_state_dict'])
        if payload.get('critic_optim_state_dict'):
            trainer.critic_optim.load_state_dict(payload['critic_optim_state_dict'])
        resume_step = payload['step']
        print(f'[train] resumed at step {resume_step:,}')
    else:
        trainer = MAPPOTrainer(train_cfg)
        resume_step = 0

    # 6. MetricWriter (TensorBoard + JSONL)
    tb_port = args.port or 6006
    writer = MetricWriter(
        logdir=run_dir,
        name='tensorboard',
        autostart_server=(args.port is not None),
        port=tb_port,
        overwrite=(resume_path is None),
        log='metrics',
    )

    # 7. Log effective config
    print(f'[train] config : {configs["_paths"]["train"]}')
    print(f'[train] env    : {configs["_paths"]["env"]}')
    print(f'[train] model  : {configs["_paths"]["model"]}')
    print(f'[train] device : {train_cfg.device}')
    print(f'[train] scenario: {train_cfg.scenario}')
    print(f'[train] total steps: {train_cfg.total_timesteps:,}')
    print(f'[train] AMP    : {train_cfg.use_amp} ({train_cfg.amp_dtype})')

    # 8. Train
    total = train_cfg.total_timesteps
    if resume_step > 0:
        total = total - resume_step  # remaining steps
    trainer.train(total_timesteps=total, writer=writer, run_dir=run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
