"""Export the trained DistributedActor to ONNX for deployment.

GRALP-style export: produces ``policy.onnx`` + ``meta.json`` with
observation/action specifications and onnxruntime numerical verification.

Usage::

    python tools/export_onnx.py --ckpt runs/multi_1/baseline/checkpoints/latest.pt
    python tools/export_onnx.py --ckpt runs/multi_1/baseline -o exported/
    python tools/export_onnx.py --ckpt runs/multi_1/baseline --opset 17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Allow running standalone
_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from multi_robot_rl.multi_agent_env import LOCAL_MAP_SIZE, MAX_FRONTIERS, N_AGENTS
from multi_robot_rl.networks import DistributedActor
from multi_robot_rl.run_manager import load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Export actor policy to ONNX.')
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to checkpoint (.pt) or run directory.')
    p.add_argument('-o', '--out-dir', type=str, default='exported',
                   help='Output directory (default: exported/).')
    p.add_argument('--opset', type=int, default=17,
                   help='ONNX opset version (default: 17).')
    p.add_argument('--device', type=str, default='cpu',
                   help='Device to load checkpoint (default: cpu).')
    p.add_argument('--verify', action='store_true', default=True,
                   help='Run onnxruntime verification (default: on).')
    p.add_argument('--no-verify', action='store_false', dest='verify',
                   help='Skip onnxruntime verification.')
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


def make_dummy_inputs(n_teammates: int = 2,
                      max_frontiers: int = MAX_FRONTIERS,
                      map_size: int = LOCAL_MAP_SIZE) -> tuple:
    """Create dummy tensors matching the actor's forward signature."""
    local_map = torch.randn(1, 3, map_size, map_size)
    own_pose = torch.randn(1, 4)
    teammates = torch.randn(1, n_teammates, 4)
    frontiers = torch.randn(1, max_frontiers, 3)
    n_frontiers = torch.tensor([max_frontiers], dtype=torch.long)
    return local_map, own_pose, teammates, frontiers, n_frontiers


def main(argv=None):
    args = parse_args(argv)
    ckpt_path = _resolve_ckpt(args.ckpt)
    print(f'[export] Loading: {ckpt_path}')

    payload = load_checkpoint(ckpt_path, map_location=args.device)
    step = payload['step']
    print(f'[export] Step: {step}')

    # Build model
    n_teammates = N_AGENTS - 1
    model = DistributedActor(
        n_actions=MAX_FRONTIERS, n_teammates=n_teammates,
    )
    model.load_state_dict(payload['actor_state_dict'])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs and export
    dummy = make_dummy_inputs(n_teammates=n_teammates)
    onnx_path = out_dir / 'policy.onnx'

    input_names = ['local_map', 'own_pose', 'teammates', 'frontiers', 'n_frontiers']
    output_names = ['action_logits']

    dynamic_axes = {
        'local_map': {0: 'batch'},
        'own_pose': {0: 'batch'},
        'teammates': {0: 'batch'},
        'frontiers': {0: 'batch'},
        'n_frontiers': {0: 'batch'},
        'action_logits': {0: 'batch'},
    }

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f'[export] ONNX saved: {onnx_path}')

    # ── meta.json ───────────────────────────────────────────────
    meta: Dict[str, Any] = {
        'checkpoint': str(ckpt_path),
        'checkpoint_step': step,
        'model': 'DistributedActor',
        'n_actions': MAX_FRONTIERS,
        'n_teammates': n_teammates,
        'n_agents': N_AGENTS,
        'map_size': LOCAL_MAP_SIZE,
        'max_frontiers': MAX_FRONTIERS,
        'input_spec': {
            'local_map': {'shape': [1, 3, LOCAL_MAP_SIZE, LOCAL_MAP_SIZE], 'dtype': 'float32'},
            'own_pose': {'shape': [1, 4], 'dtype': 'float32'},
            'teammates': {'shape': [1, n_teammates, 4], 'dtype': 'float32'},
            'frontiers': {'shape': [1, MAX_FRONTIERS, 3], 'dtype': 'float32'},
            'n_frontiers': {'shape': [1], 'dtype': 'int64'},
        },
        'output_spec': {
            'action_logits': {'shape': [1, MAX_FRONTIERS], 'dtype': 'float32'},
        },
        'opset': args.opset,
    }
    meta_path = out_dir / 'meta.json'
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    print(f'[export] meta saved: {meta_path}')

    # ── onnxruntime verification ────────────────────────────────
    if args.verify:
        _verify(onnx_path, model, dummy)

    return 0


def _verify(onnx_path: Path, pytorch_model: torch.nn.Module,
            dummy: tuple) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print('[export] WARNING: onnx/onnxruntime not installed, skip verify')
        return

    # ONNX model check
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print('[export] ONNX model check: PASS')

    # Numerical comparison
    with torch.no_grad():
        pt_out = pytorch_model(*dummy).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_inputs = {
        'local_map': dummy[0].cpu().numpy().astype(np.float32),
        'own_pose': dummy[1].cpu().numpy().astype(np.float32),
        'teammates': dummy[2].cpu().numpy().astype(np.float32),
        'frontiers': dummy[3].cpu().numpy().astype(np.float32),
        'n_frontiers': dummy[4].cpu().numpy().astype(np.int64),
    }
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f'[export] ONNX vs PyTorch max diff: {max_diff:.2e}')

    if max_diff < 1e-5:
        print('[export] Numerical verification: PASS ✓')
    else:
        print('[export] Numerical verification: WARNING (diff > 1e-5)')


if __name__ == '__main__':
    sys.exit(main())
