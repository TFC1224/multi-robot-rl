"""Run directory management with GRALP-style timestamped directories.

Provides::

    RunManager  — create, find, and list training runs
    save_checkpoint   — unified checkpoint format (step-N.pt + latest.pt + meta.json)
    load_checkpoint   — load a unified checkpoint
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ── path patterns ────────────────────────────────────────────────

_STEP_CKPT_RE = re.compile(r'^step-(\d+)\.pt$')
_RUN_DIR_RE = re.compile(r'^(\d{8}-\d{6})(?:-(.+))?$')


# ── RunManager ───────────────────────────────────────────────────

class RunManager:
    """Manage timestamped training run directories.

    Each run directory is::

        runs/{scenario}/{YYYYMMDD-HHMMSS}[-{tag}]/
          config_snapshot/
            train_config.json
            env_config.json
            model_config.json
          checkpoints/
            step-50000.pt
            step-100000.pt
            ...
            latest.pt
            meta.json
          tensorboard/
          metrics.jsonl

    Args:
        runs_root: Root directory for all runs (default ``runs/``).
        scenario: Sub-directory grouping runs by scenario name.
    """

    def __init__(self, runs_root: str | Path = 'runs',
                 scenario: str = 'default') -> None:
        self._root = Path(runs_root)
        self._scenario = str(scenario)

    @property
    def root(self) -> Path:
        return self._root

    # ── run creation ─────────────────────────────────────────

    def create_run(self, tag: str | None = None,
                    configs: Dict[str, Any] | None = None) -> Path:
        """Create a new timestamped run directory and snapshot configs.

        Returns the run directory path.
        """
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = f'{ts}-{tag}' if tag else ts
        run_dir = self._root / self._scenario / name
        run_dir.mkdir(parents=True, exist_ok=False)

        (run_dir / 'checkpoints').mkdir()
        (run_dir / 'tensorboard').mkdir()

        if configs is not None:
            self.snapshot_configs(run_dir, configs)

        print(f'[run] created: {run_dir}')
        return run_dir

    def snapshot_configs(self, run_dir: str | Path,
                          configs: Dict[str, Any]) -> None:
        """Copy the current effective configs into the run directory."""
        snap_dir = Path(run_dir) / 'config_snapshot'
        snap_dir.mkdir(parents=True, exist_ok=True)

        for name in ('train', 'env', 'model'):
            cfg = configs.get(name)
            if cfg is None:
                continue
            out = snap_dir / f'{name}_config.json'
            out.write_text(
                json.dumps(cfg, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8')

    # ── checkpoint utilities ─────────────────────────────────

    def checkpoint_dir(self, run_dir: str | Path) -> Path:
        return Path(run_dir) / 'checkpoints'

    # ── run discovery ────────────────────────────────────────

    def find_run_by_tag(self, tag: str) -> Optional[Path]:
        """Find the most recent run directory matching *tag*."""
        scenario_dir = self._root / self._scenario
        if not scenario_dir.is_dir():
            return None
        matches = []
        for d in scenario_dir.iterdir():
            m = _RUN_DIR_RE.match(d.name)
            if not m or m.group(2) != tag:
                continue
            if d.is_dir():
                matches.append(d)
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def list_runs(self) -> List[Path]:
        """List all run directories for the current scenario, newest first."""
        scenario_dir = self._root / self._scenario
        if not scenario_dir.is_dir():
            return []
        dirs = [d for d in scenario_dir.iterdir()
                if d.is_dir() and _RUN_DIR_RE.match(d.name)]
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs

    def get_latest_checkpoint(self, run_dir: str | Path) -> Optional[Path]:
        """Return the path to ``latest.pt`` if it exists."""
        ckpt_dir = self.checkpoint_dir(run_dir)
        latest = ckpt_dir / 'latest.pt'
        return latest if latest.is_file() else None

    def find_step_checkpoint(self, run_dir: str | Path,
                              step: int) -> Optional[Path]:
        """Return the checkpoint for a specific step number."""
        ckpt = self.checkpoint_dir(run_dir) / f'step-{step}.pt'
        return ckpt if ckpt.is_file() else None


# ── unified checkpoint format ────────────────────────────────────

def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    step: int,
    actor_state: Dict[str, Any],
    critic_state: Dict[str, Any],
    actor_optim: Dict[str, Any] | None = None,
    critic_optim: Dict[str, Any] | None = None,
    scaler_state: Dict[str, Any] | None = None,
    actor_scheduler: Dict[str, Any] | None = None,
    critic_scheduler: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
    metrics: Dict[str, float] | None = None,
) -> Path:
    """Save a unified checkpoint containing model weights + training state.

    Produces:
      ``step-{step}.pt``  — the full checkpoint
      ``latest.pt``       — copy of the latest checkpoint
      ``meta.json``       — lightweight metadata for external tooling
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        'step': int(step),
        'actor_state_dict': actor_state,
        'critic_state_dict': critic_state,
    }
    if actor_optim is not None:
        payload['actor_optim_state_dict'] = actor_optim
    if critic_optim is not None:
        payload['critic_optim_state_dict'] = critic_optim
    if scaler_state is not None:
        payload['scaler_state_dict'] = scaler_state
    if actor_scheduler is not None:
        payload['actor_scheduler_state_dict'] = actor_scheduler
    if critic_scheduler is not None:
        payload['critic_scheduler_state_dict'] = critic_scheduler
    if config is not None:
        payload['config'] = config

    # Save numbered checkpoint
    ckpt_path = ckpt_dir / f'step-{int(step)}.pt'
    torch.save(payload, ckpt_path)

    # Update latest.pt (copy, not symlink — safer on Windows)
    latest_path = ckpt_dir / 'latest.pt'
    try:
        shutil.copy2(ckpt_path, latest_path)
    except OSError:
        pass

    # Write meta.json
    meta: Dict[str, Any] = {
        'step': int(step),
        'timestamp': datetime.now().isoformat(),
    }
    if metrics is not None:
        meta['metrics'] = {k: float(v) for k, v in metrics.items()}
    meta_path = ckpt_dir / 'meta.json'
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    return ckpt_path


def load_checkpoint(
    ckpt_path: str | Path,
    *,
    map_location: str = 'cpu',
    load_optimizer: bool = True,
) -> Dict[str, Any]:
    """Load a unified checkpoint.

    Returns the full payload dict.  Caller is responsible for extracting
    ``actor_state_dict``, ``critic_state_dict``, etc. and restoring them
    into the respective objects.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    required = {'step', 'actor_state_dict'}
    missing = required - set(payload.keys())
    if missing:
        raise KeyError(f'checkpoint missing required keys: {missing}')

    return payload
