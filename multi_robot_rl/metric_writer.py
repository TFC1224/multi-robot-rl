"""MetricWriter: TensorBoard SummaryWriter + optional JSONL text log.

Adapted from GRALP's ``rl_ppo/writer.py``.  Provides a singleton-style
writer that auto-starts a TensorBoard server subprocess and supports
dual-channel logging (TensorBoard events + JSONL).

Usage::

    from multi_robot_rl.metric_writer import MetricWriter

    writer = MetricWriter(logdir='runs/my_run', name='tb', port=6006)
    writer.log({'actor_loss': 0.5, 'coverage': 0.8}, step=1000, scope='train')
    writer.flush()
    writer.close()
"""

from __future__ import annotations

import atexit
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, Mapping


def _find_tensorboard() -> bool:
    """Check whether tensorboard is importable."""
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


class MetricWriter:
    """Writes scalar metrics to TensorBoard and optionally to a JSONL log.

    Args:
        logdir: Root directory for TensorBoard logs.
        name: Run sub-directory name (TB UI run identifier).
        autostart_server: Whether to auto-launch ``tensorboard`` subprocess.
        port: Web server port (default 6006).
        host: Web server listen address (default ``0.0.0.0``).
        flush_every_s: SummaryWriter async flush interval (seconds).
        overwrite: Purge old events in the run directory on init.
        log: If non-empty, enable JSONL text log at ``{log}.jsonl``.
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        name: str | None = None,
        autostart_server: bool = True,
        port: int = 6006,
        host: str = '0.0.0.0',
        flush_every_s: int = 10,
        overwrite: bool = False,
        log: str | None = None,
    ) -> None:
        self._logdir = Path(logdir)
        self._events_dir = self._logdir / name if name else self._logdir
        self._events_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._purge_events()

        self._writer = None
        if _find_tensorboard():
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(
                log_dir=str(self._events_dir), flush_secs=int(flush_every_s))
        else:
            sys.stderr.write(
                '[metric_writer] tensorboard not installed; '
                'TB logging disabled (JSONL only)\n')

        self._lock = threading.Lock()
        self._log_file: IO[str] | None = None
        if log:
            log_path = self._events_dir / f'{log}.jsonl'
            self._log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            sys.stdout.write(f'[metric_writer] JSONL log: {log_path}\n')

        self._proc: subprocess.Popen | None = None
        self._port = int(port)
        self._host = str(host)
        if autostart_server and _find_tensorboard():
            self._start_server()

        atexit.register(self.close)

    # ── internal helpers ─────────────────────────────────────────

    def _purge_events(self) -> None:
        removed = 0
        for f in self._events_dir.iterdir():
            if f.is_file() and f.name.startswith('events.out.tfevents'):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            sys.stdout.write(
                f'[metric_writer] purged {removed} old event files '
                f'({self._events_dir})\n')

    def _start_server(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, '-m', 'tensorboard.main',
                 '--logdir', str(self._logdir),
                 '--port', str(self._port),
                 '--host', self._host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as exc:
            sys.stderr.write(
                f'[metric_writer] WARNING: could not start tensorboard '
                f'({exc!r})\n')
            self._proc = None
            return

        display_host = 'localhost' if self._host in ('0.0.0.0', '') else self._host
        sys.stdout.write(
            f'[metric_writer] TensorBoard starting: '
            f'http://{display_host}:{self._port} '
            f'(logdir={self._logdir})\n')
        sys.stdout.flush()

        threading.Thread(
            target=self._wait_for_ready,
            args=(60.0, display_host),
            daemon=True,
        ).start()

    def _wait_for_ready(self, timeout_s: float, display_host: str) -> None:
        probe_host = '127.0.0.1' if self._host in ('0.0.0.0', '') else self._host
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((probe_host, self._port), timeout=0.3):
                    sys.stdout.write(
                        f'[metric_writer] TensorBoard ready: '
                        f'http://{display_host}:{self._port}\n')
                    sys.stdout.flush()
                    return
            except OSError:
                time.sleep(0.5)
        sys.stderr.write(
            f'[metric_writer] WARNING: TensorBoard not ready within '
            f'{int(timeout_s)}s (port={self._port})\n')

    # ── public API ───────────────────────────────────────────────

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: int,
        scope: str | None = None,
    ) -> None:
        """Write a batch of scalar metrics.

        Args:
            metrics: Name → value mapping.
            step: Global training step.
            scope: Tag prefix for TensorBoard grouping (e.g. ``'train'``).
        """
        prefix = f'{scope}/' if scope else ''
        step_i = int(step)
        coerced: Dict[str, float] = {name: float(value) for name, value in metrics.items()}

        with self._lock:
            if self._writer is not None:
                for name, value in coerced.items():
                    self._writer.add_scalar(f'{prefix}{name}', value, step_i)

            if self._log_file is not None:
                record: Dict[str, Any] = {
                    'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'scope': scope or '',
                    'step': step_i,
                    'metrics': coerced,
                }
                self._log_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    def write_hparams(self, hparams: Mapping[str, Any],
                       metrics: Mapping[str, float]) -> None:
        """Log hyperparameters for TensorBoard HParams dashboard."""
        if self._writer is not None:
            self._writer.add_hparams(
                {k: v for k, v in hparams.items()},
                {k: float(v) for k, v in metrics.items()})

    def flush(self) -> None:
        """Flush pending writes to disk."""
        with self._lock:
            if self._writer is not None:
                self._writer.flush()
            if self._log_file is not None:
                self._log_file.flush()

    def close(self) -> None:
        """Close the writer and terminate the TensorBoard subprocess."""
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.flush()
                    self._writer.close()
                finally:
                    self._writer = None
            if self._log_file is not None:
                try:
                    self._log_file.close()
                finally:
                    self._log_file = None

        if self._proc is not None:
            proc = self._proc
            self._proc = None
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
