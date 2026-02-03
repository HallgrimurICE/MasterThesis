from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_ACTIVE_RECORDER: Optional["TimingRecorder"] = None


def set_timing_recorder(recorder: Optional["TimingRecorder"]) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder


def get_timing_recorder() -> Optional["TimingRecorder"]:
    return _ACTIVE_RECORDER


@contextmanager
def timer(name: str):
    recorder = _ACTIVE_RECORDER
    if recorder is None or not recorder.enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        recorder.record(name, time.perf_counter() - start)


@dataclass
class GpuLogger:
    output_dir: Path
    interval: int = 5
    enabled: bool = True
    _nvidia_smi: Optional[str] = field(default=None, init=False)
    _log_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self._nvidia_smi = shutil.which("nvidia-smi")
        if self._nvidia_smi is None:
            self.enabled = False
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / "gpu.log"
        self.log_once(tag="start")

    def log_once(self, *, tag: str) -> None:
        if not self.enabled or self._nvidia_smi is None or self._log_path is None:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        info_lines = [
            f"[{timestamp}] {tag}",
        ]
        try:
            result = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout:
                info_lines.extend(result.stdout.strip().splitlines())
        except OSError:
            return

        torch_info = _torch_device_info()
        if torch_info:
            info_lines.append(torch_info)

        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(info_lines) + "\n")

    def maybe_log(self, round_index: int) -> None:
        if not self.enabled or self.interval <= 0:
            return
        if round_index % self.interval == 0:
            self.log_once(tag=f"round={round_index}")


@dataclass
class TimingRecorder:
    output_dir: Path
    enabled: bool = True
    round_index: Optional[int] = None
    round_totals: Dict[str, float] = field(default_factory=dict)
    totals: Dict[str, float] = field(default_factory=dict)
    rounds: list[Dict[str, float]] = field(default_factory=list)
    gpu_logger: Optional[GpuLogger] = None

    def start_round(self, round_index: int) -> None:
        if not self.enabled:
            return
        self.round_index = round_index
        self.round_totals = {}

    def record(self, name: str, elapsed: float) -> None:
        if not self.enabled:
            return
        self.round_totals[name] = self.round_totals.get(name, 0.0) + elapsed
        self.totals[name] = self.totals.get(name, 0.0) + elapsed

    def end_round(self) -> None:
        if not self.enabled or self.round_index is None:
            return
        summary = " ".join(
            f"{name}={duration:.4f}s" for name, duration in sorted(self.round_totals.items())
        )
        print(f"[timing] round={self.round_index} {summary}")
        self.rounds.append({"round": float(self.round_index), **self.round_totals})
        if self.gpu_logger is not None:
            self.gpu_logger.maybe_log(self.round_index)
        self.round_index = None

    def finalize(self) -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "totals": self.totals,
            "rounds": self.rounds,
        }
        with (self.output_dir / "timing.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def _torch_device_info() -> str:
    if os.environ.get("DISABLE_TORCH_INFO") == "1":
        return ""
    if importlib.util.find_spec("torch") is None:
        return ""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = str(torch.get_default_dtype())
    autocast = torch.is_autocast_enabled()
    return f"torch device={device} dtype={dtype} autocast={autocast}"


__all__ = ["GpuLogger", "TimingRecorder", "get_timing_recorder", "set_timing_recorder", "timer"]
