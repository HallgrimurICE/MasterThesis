from __future__ import annotations

import argparse
import cProfile
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

from .demo import (
    run_standard_board_br_vs_neg,
    run_standard_board_mixed_tom_demo,
    run_standard_board_with_random_agents,
)
from .timing import GpuLogger, TimingRecorder, set_timing_recorder
from .types import Power


def _parse_powers(raw: Optional[str]) -> List[Power]:
    if not raw:
        return []
    return [Power(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_seeds(raw: Optional[str], num_seeds: Optional[int], default_seed: int) -> List[int]:
    if raw:
        raw = raw.strip()
        if ":" in raw:
            start_str, end_str = raw.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            return list(range(start, end + 1))
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    if num_seeds is not None:
        return list(range(num_seeds))
    return [default_seed]


def _resolve_seeds(args: argparse.Namespace) -> List[int]:
    seeds = _parse_seeds(args.seeds, args.num_seeds, args.seed)
    job_index = os.getenv("LSB_JOBINDEX")
    if job_index is None:
        return seeds
    job_seed = int(job_index)
    if seeds:
        idx = job_seed - 1
        if 0 <= idx < len(seeds):
            return [seeds[idx]]
    return [job_seed]


def _run_mode(args: argparse.Namespace, seed: int) -> None:
    if args.mode == "random_agents":
        run_standard_board_with_random_agents(
            rounds=args.rounds,
            visualize=args.visualize,
            seed=seed,
        )
        return
    if args.mode == "br_vs_neg":
        run_standard_board_br_vs_neg(
            weights_path=args.weights_path,
            rounds=args.rounds,
            seed=seed,
            k_candidates=args.k_candidates,
            action_rollouts=args.action_rollouts,
            rss_rollouts=args.rss_rollouts,
            tom_depth=args.tom_depth,
            negotiation_powers=_parse_powers(args.negotiation_powers),
            baseline_powers=_parse_powers(args.baseline_powers),
            use_heuristic_value=args.use_heuristic_value,
            stop_on_winner=not args.disable_stop_on_winner,
            visualize=args.visualize,
        )
        return
    if args.mode == "mixed_tom":
        run_standard_board_mixed_tom_demo(
            weights_path=args.weights_path,
            rounds=args.rounds,
            seed=seed,
            k_candidates=args.k_candidates,
            action_rollouts=args.action_rollouts,
            rss_rollouts=args.rss_rollouts,
            negotiation_powers=_parse_powers(args.negotiation_powers),
            default_tom_depth=args.tom_depth,
            use_heuristic_value=args.use_heuristic_value,
            stop_on_winner=not args.disable_stop_on_winner,
            visualize=args.visualize,
        )
        return
    raise ValueError(f"Unknown mode: {args.mode}")


def _maybe_torch_profiler(args: argparse.Namespace):
    if not args.torch_profiler:
        return None
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        print("[profile] torch not available; skipping torch profiler.")
        return None
    import torch

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def _ensure_run_dir(output_root: str, exp_name: str, seed: int) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / exp_name / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp_dir = run_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    return timestamp_dir


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Diplomacy experiment runner")
    parser.add_argument("--mode", required=True, choices=["random_agents", "br_vs_neg", "mixed_tom"])
    parser.add_argument("--weights_path", default="", help="Path to sl_params.npz for DeepMind modes")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Seed list (e.g. 0,1,2) or range (0:99)")
    parser.add_argument("--num_seeds", type=int, default=None, help="Generate seeds 0..N-1")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--output_root", type=str, default="runs")
    parser.add_argument("--k_candidates", type=int, default=1)
    parser.add_argument("--action_rollouts", type=int, default=1)
    parser.add_argument("--rss_rollouts", type=int, default=1)
    parser.add_argument("--tom_depth", type=int, default=1)
    parser.add_argument("--negotiation_powers", type=str, default="")
    parser.add_argument("--baseline_powers", type=str, default="")
    parser.add_argument("--use_heuristic_value", action="store_true")
    parser.add_argument("--disable_stop_on_winner", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--torch_profiler", action="store_true")
    parser.add_argument("--gpu_log_interval", type=int, default=5)
    parser.add_argument("--inference_batch_size", type=int, default=16)

    args = parser.parse_args(list(argv) if argv is not None else None)
    os.environ["INFERENCE_BATCH_SIZE"] = str(args.inference_batch_size)

    seeds = _resolve_seeds(args)
    for seed in seeds:
        run_dir = _ensure_run_dir(args.output_root, args.exp_name, seed)
        recorder = None
        if args.profile:
            gpu_logger = GpuLogger(run_dir, interval=args.gpu_log_interval, enabled=True)
            recorder = TimingRecorder(output_dir=run_dir, enabled=True, gpu_logger=gpu_logger)
        set_timing_recorder(recorder)

        def run_once() -> None:
            _run_mode(args, seed)

        profiler = _maybe_torch_profiler(args)
        if args.cprofile:
            profile = cProfile.Profile()
            profile.enable()
            if profiler is not None:
                with profiler:
                    run_once()
                profiler.export_chrome_trace(str(run_dir / "torch_profile.json"))
            else:
                run_once()
            profile.disable()
            profile.dump_stats(str(run_dir / "cprofile.prof"))
        else:
            if profiler is not None:
                with profiler:
                    run_once()
                profiler.export_chrome_trace(str(run_dir / "torch_profile.json"))
            else:
                run_once()

        if recorder is not None:
            recorder.finalize()
        set_timing_recorder(None)


if __name__ == "__main__":
    main()
