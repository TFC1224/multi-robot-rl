"""Offline evaluation script for trained MAPPO policies.

Loads a unified checkpoint and runs deterministic evaluation in the
Gymnasium environment, reporting coverage, fairness, and collision metrics.

Usage::

    python scripts/eval_mappo.py --ckpt runs/multi_1/baseline/checkpoints/latest.pt
    python scripts/eval_mappo.py --ckpt runs/multi_1/baseline --n_episodes 20 --seed 42
    python scripts/eval_mappo.py --ckpt runs/multi_1/baseline --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Allow running this script standalone without installing the ROS 2 package.
_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from multi_robot_rl.multi_agent_env import (
    MAX_FRONTIERS, N_AGENTS, EnvConfig, MultiAgentExplorerEnv,
)
from multi_robot_rl.networks import DistributedActor
from multi_robot_rl.run_manager import load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Evaluate a trained MAPPO policy.')
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to a checkpoint file or run directory.')
    p.add_argument('--n_episodes', type=int, default=10,
                   help='Number of evaluation episodes (default: 10).')
    p.add_argument('--scenario', type=str, default='multi_1',
                   choices=['multi_1', 'multi_2', 'multi_3'])
    p.add_argument('--seed', type=int, default=0,
                   help='Evaluation random seed.')
    p.add_argument('--device', type=str, default='cpu',
                   help='Inference device.')
    p.add_argument('--json', type=str, default=None,
                   help='If set, write summary JSON to this path.')
    p.add_argument('--render', action='store_true',
                   help='Print per-step coverage in real time.')
    return p.parse_args(argv)


def _resolve_ckpt(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        ckpt_dir = p / 'checkpoints'
        latest = ckpt_dir / 'latest.pt'
        if latest.exists():
            return latest
        candidates = sorted(ckpt_dir.glob('step-*.pt'))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
    raise FileNotFoundError(f'Checkpoint not found: {path}')


def run_episode(env: MultiAgentExplorerEnv, model: DistributedActor,
                device: torch.device, render: bool = False) -> Dict[str, Any]:
    """Run one deterministic episode. Returns summary dict."""
    obs_tuple, _ = env.reset()
    total_reward = 0.0
    steps = 0
    collisions = 0
    coverages = []

    while True:
        observations = list(obs_tuple)
        local_maps = np.stack([o['local_map'] for o in observations], axis=0)
        own_poses = np.stack([o['own_pose'] for o in observations], axis=0)
        teammates = np.stack([o['teammates'] for o in observations], axis=0)
        frontiers = np.stack([o['frontiers'] for o in observations], axis=0)
        n_frontiers = np.array([o['n_frontiers'] for o in observations],
                               dtype=np.int64)

        with torch.no_grad():
            local_maps_t = torch.from_numpy(local_maps).to(device)
            own_poses_t = torch.from_numpy(own_poses).to(device)
            teammates_t = torch.from_numpy(teammates).to(device)
            frontiers_t = torch.from_numpy(frontiers).to(device)
            n_frontiers_t = torch.from_numpy(n_frontiers).to(device)
            logits = model(local_maps_t, own_poses_t, teammates_t,
                           frontiers_t, n_frontiers_t)
            actions = torch.argmax(logits, dim=-1).cpu().numpy()

        obs_tuple, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(sum(rewards))
        steps += 1
        collisions += 1 if info['collision_count'] > collisions else 0
        coverages.append(float(info['team_coverage']))

        if render:
            print(f'  step={steps:>4d}  '
                  f'coverage={info["team_coverage"]:.4f}  '
                  f'jain={info["jain_index"]:.4f}  '
                  f'overlap={info["overlap_ratio"]:.4f}')

        if terminated or truncated:
            break

    return {
        'steps': steps,
        'total_reward': total_reward,
        'final_coverage': coverages[-1],
        'max_coverage': max(coverages),
        'collisions': collisions,
        'terminated': bool(terminated),
        'jain_final': float(info['jain_index']),
        'overlap_ratio': float(info['overlap_ratio']),
        'coverage_trace': coverages,
    }


def main(argv=None):
    args = parse_args(argv)
    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f'[eval] Loading checkpoint: {ckpt_path}')

    payload = load_checkpoint(ckpt_path, map_location=args.device)
    print(f'[eval] Checkpoint step: {payload["step"]}')

    env = MultiAgentExplorerEnv(EnvConfig(
        scenario=args.scenario,
        target_coverage=0.95,
        max_steps_per_episode=300,
    ))

    device = torch.device(args.device)
    model = DistributedActor(
        n_actions=MAX_FRONTIERS, n_teammates=N_AGENTS - 1,
    ).to(device)
    model.load_state_dict(payload['actor_state_dict'])
    model.eval()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    print(f'[eval] Running {args.n_episodes} episodes (scenario={args.scenario})')

    for ep in range(args.n_episodes):
        ep_result = run_episode(env, model, device, render=args.render)
        ep_result['episode'] = ep
        results.append(ep_result)
        print(f'  ep={ep:>3d}  steps={ep_result["steps"]:>3d}  '
              f'coverage={ep_result["final_coverage"]:.4f}  '
              f'jain={ep_result["jain_final"]:.4f}  '
              f'collisions={ep_result["collisions"]}  '
              f'done={ep_result["terminated"]}')

    # ── summary stats ──────────────────────────────────────────
    coverages = [r['final_coverage'] for r in results]
    jains = [r['jain_final'] for r in results]
    steps_list = [r['steps'] for r in results]
    collisions_list = [r['collisions'] for r in results]
    success_rate = sum(1 for r in results if r['terminated']) / len(results)

    summary = {
        'checkpoint': str(ckpt_path),
        'checkpoint_step': payload['step'],
        'scenario': args.scenario,
        'n_episodes': args.n_episodes,
        'seed': args.seed,
        'coverage_mean': float(np.mean(coverages)),
        'coverage_std': float(np.std(coverages)),
        'jain_mean': float(np.mean(jains)),
        'jain_std': float(np.std(jains)),
        'steps_mean': float(np.mean(steps_list)),
        'steps_std': float(np.std(steps_list)),
        'collisions_mean': float(np.mean(collisions_list)),
        'collisions_std': float(np.std(collisions_list)),
        'success_rate': success_rate,
        'per_episode': results,
    }

    print(f'\n[eval] ── Summary ──')
    print(f'  Coverage : {summary["coverage_mean"]:.4f} ± {summary["coverage_std"]:.4f}')
    print(f'  Jain     : {summary["jain_mean"]:.4f} ± {summary["jain_std"]:.4f}')
    print(f'  Steps    : {summary["steps_mean"]:.1f} ± {summary["steps_std"]:.1f}')
    print(f'  Collision: {summary["collisions_mean"]:.2f} ± {summary["collisions_std"]:.2f}')
    print(f'  Success  : {summary["success_rate"]:.1%}')

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                       encoding='utf-8')
        print(f'[eval] Results written to {out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
