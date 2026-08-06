"""Configuration loader with JSON file support and CLI override merging.

Replaces the previous dead ``mappo_config.yaml`` with a GRALP-inspired
three-tier JSON config system:

* ``train_config.json``  — PPO hyperparameters, logging, checkpointing
* ``env_config.json``    — environment, observation, reward settings
* ``model_config.json``  — network architecture parameters
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


_JSON_COMMENT_RE = re.compile(r'//.*$', re.MULTILINE)


def load_json_config(path: str | Path | None) -> Dict[str, Any]:
    """Load a JSON config file, stripping ``//`` comments.

    Returns an empty dict when *path* is ``None`` or the file does not exist.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text(encoding='utf-8')
    text = _JSON_COMMENT_RE.sub('', text)
    return json.loads(text)


def resolve_config_path(train_cfg_path: str | Path, key: str,
                        train_cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve a sibling config path relative to the train config file."""
    val = train_cfg.get(key)
    if not val:
        return None
    if Path(val).is_absolute():
        return val
    return str(Path(train_cfg_path).resolve().parent / val)


def load_all_configs(train_cfg_path: str | Path,
                     env_cfg_path: Optional[str | Path] = None,
                     model_cfg_path: Optional[str | Path] = None,
                     ) -> Dict[str, Any]:
    """Load all three configs and return a merged dictionary.

    Resolution order:
    1. Explicit paths passed as arguments
    2. Paths referenced in ``train_config.json`` (``env_config`` / ``model_config`` keys)
    3. ``config/env_config.json`` / ``config/model_config.json`` relative to cwd
    """
    train_cfg = load_train_config(train_cfg_path)
    train_path = Path(train_cfg_path)

    # Resolve env config path
    if env_cfg_path is None:
        env_cfg_path = resolve_config_path(train_path, 'env_config', train_cfg)
    if env_cfg_path is None:
        env_cfg_path = train_path.parent / 'env_config.json'
    env_cfg = load_json_config(env_cfg_path)

    # Resolve model config path
    if model_cfg_path is None:
        model_cfg_path = resolve_config_path(train_path, 'model_config', train_cfg)
    if model_cfg_path is None:
        model_cfg_path = train_path.parent / 'model_config.json'
    model_cfg = load_json_config(model_cfg_path)

    return {
        'train': train_cfg,
        'env': env_cfg,
        'model': model_cfg,
        '_paths': {
            'train': str(train_path),
            'env': str(env_cfg_path),
            'model': str(model_cfg_path),
        },
    }


def load_train_config(path: str | Path | None) -> Dict[str, Any]:
    """Load the training config and fill in defaults (GRALP-style)."""
    raw = load_json_config(path) if path else {}
    cfg: Dict[str, Any] = {}

    cfg.setdefault('device', 'auto')
    cfg.setdefault('env_config', 'env_config.json')
    cfg.setdefault('model_config', 'model_config.json')

    cfg.setdefault('algorithm', 'MAPPO')
    cfg.setdefault('total_timesteps', 1_000_000)
    cfg.setdefault('n_steps', 2048)
    cfg.setdefault('n_epochs', 10)
    cfg.setdefault('batch_size', 64)
    cfg.setdefault('gamma', 0.99)
    cfg.setdefault('gae_lambda', 0.95)
    cfg.setdefault('clip_range', 0.2)
    cfg.setdefault('entropy_coef', 0.01)
    cfg.setdefault('value_loss_coef', 0.5)
    cfg.setdefault('max_grad_norm', 0.5)
    cfg.setdefault('actor_lr', 3e-4)
    cfg.setdefault('critic_lr', 5e-4)

    cfg.setdefault('amp', False)
    cfg.setdefault('amp_dtype', 'bfloat16')
    cfg.setdefault('lr_schedule', 'none')
    cfg.setdefault('value_norm', False)
    cfg.setdefault('ortho_init', False)
    cfg.setdefault('ortho_gain', 0.01)

    cfg.setdefault('save_dir', 'models')
    cfg.setdefault('save_freq', 50_000)
    cfg.setdefault('log_freq', 2_048)
    cfg.setdefault('seed', 0)

    # Merge user values (they take precedence over defaults)
    cfg.update({k: v for k, v in raw.items() if k not in cfg or v != cfg[k]})
    # Actually, raw values should override defaults:
    cfg.update(raw)

    return cfg


def merge_cli_overrides(config: Dict[str, Any],
                         overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI overrides into the merged config dict.

    Only overrides keys that were explicitly provided (non-None values).
    Nested keys use ``.`` notation (e.g. ``env.scenario``).
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if '.' in key:
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        else:
            # Try to distribute to the right sub-config
            _auto_route_override(config, key, value)
    return config


def _auto_route_override(config: Dict[str, Any], key: str, value: Any) -> None:
    """Route a flat CLI key to the correct sub-config via heuristic."""
    # Known env keys
    if key in ('scenario', 'n_agents', 'target_coverage', 'max_steps_per_episode',
               'max_frontiers', 'local_map_size', 'step_cells', 'sensor_range_m',
               'collision_threshold_cells'):
        config.setdefault('env', {})[key] = value
        return
    # Known model keys
    if key in ('ortho_init', 'ortho_gain', 'activation'):
        config.setdefault('model', {})[key] = value
        return
    # Everything else goes to train config
    config.setdefault('train', {})[key] = value


def save_config_snapshot(run_dir: str | Path,
                          configs: Dict[str, Any]) -> None:
    """Save the current config snapshot into ``<run_dir>/config_snapshot/``."""
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
