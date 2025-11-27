"""Demo for sampling a random Diplomacy board and visualizing value probabilities.

This script mirrors the logic from ``tests/test_value_visualization.py`` but is set
up for interactive/demo usage. It will try to load a pretrained Supervised Learning
(SL) checkpoint (``policytraining/data/fppi2.npz`` preferred, falling back to
``policytraining/data/sl_params.npz``) and otherwise initialize random parameters
for the network.

Usage
-----
$ python value_visualization_demo.py --seed 123 --save random_state.png

If matplotlib/networkx are installed, a mesh visualization will be shown (or
saved if ``--save`` is provided). The script also prints the per-power value
probabilities to stdout.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
POLICYTRAINING_DIR = PROJECT_ROOT / "policytraining"

for path in (SRC_DIR, POLICYTRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:  # noqa: E402
    import numpy as np
    import jax
    import jax.numpy as jnp
    from diplomacy.deepmind.actions import legal_actions_from_state
    from diplomacy.deepmind.build_observation import build_observation
    from diplomacy.maps import standard_board
    from diplomacy.state import GameState
    from diplomacy.types import Power, ProvinceType, Unit, UnitType
    from diplomacy.viz.mesh import MATPLOTLIB_AVAILABLE, NX_AVAILABLE, visualize_state_mesh
    from policytraining.network import config as net_config
    from policytraining.network import network_policy
    from policytraining.network import parameter_provider
    from policytraining.run_sl import make_sl_policy
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    missing_dep = exc.name or "a required dependency"
    raise SystemExit(f"Missing dependency '{missing_dep}'. Please install project requirements.") from exc


_STANDARD_POWERS: Tuple[Power, ...] = (
    Power("Austria"),
    Power("England"),
    Power("France"),
    Power("Germany"),
    Power("Italy"),
    Power("Russia"),
    Power("Turkey"),
)


class _RandomParameterProvider:
    """Generate random parameters/state for the Diplomacy network."""

    def __init__(self, network_cls, network_kwargs, rng_seed: int = 0):
        rng = jax.random.PRNGKey(rng_seed)
        params, net_state = network_cls.initial_inference_params_and_state(
            network_kwargs, rng, network_kwargs["num_players"]
        )
        self._params = params
        self._state = net_state
        self._step = jnp.array(0)

    def params_for_actor(self):
        return self._params, self._state, self._step


def _random_standard_state(seed: int = 0) -> GameState:
    rng = random.Random(seed)
    board = standard_board()
    province_names = list(board)
    rng.shuffle(province_names)

    units = {}
    for power, loc in zip(_STANDARD_POWERS, province_names):
        province = board[loc]
        if province.province_type == ProvinceType.SEA:
            unit_type = UnitType.FLEET
        elif province.province_type == ProvinceType.COAST:
            unit_type = rng.choice([UnitType.ARMY, UnitType.FLEET])
        else:
            unit_type = UnitType.ARMY
        units[loc] = Unit(power=power, loc=loc, unit_type=unit_type)

    return GameState(board=board, units=units, powers=set(_STANDARD_POWERS))


def _load_policy(checkpoint_path: str | None = None, rng_seed: int = 0):
    checkpoint_candidates = [
        Path(checkpoint_path) if checkpoint_path else None,
        Path("policytraining/data/fppi2.npz"),
        Path("policytraining/data/sl_params.npz"),
    ]

    for params_path in checkpoint_candidates:
        if params_path is not None and params_path.is_file():
            policy = make_sl_policy(str(params_path))
            print(f"Loaded pretrained checkpoint from: {params_path}")
            return policy

    cfg = net_config.get_config()
    network_cls = cfg.network_class
    # Use training mode when seeding random parameters so that batch norm
    # statistics are initialized. The handler below will use the same
    # configuration for inference.
    network_kwargs = dict(cfg.network_kwargs, is_training=True)
    provider = _RandomParameterProvider(network_cls, network_kwargs, rng_seed)
    handler = parameter_provider.SequenceNetworkHandler(
        network_cls=network_cls,
        network_config=network_kwargs,
        rng_seed=rng_seed,
        parameter_provider=provider,
    )
    handler.reset()
    print("No checkpoint found; using randomly initialized parameters.")
    return network_policy.Policy(
        network_handler=handler,
        num_players=network_kwargs["num_players"],
        temperature=0.1,
        calculate_all_policies=False,
    )


def _evaluate_values(state: GameState, policy) -> Sequence[float]:
    observation = build_observation(state, last_actions=[])
    legal_actions = legal_actions_from_state(state)
    slots_list = list(range(len(legal_actions)))
    _, info = policy.actions(
        slots_list=slots_list,
        observation=observation,
        legal_actions=legal_actions,
    )
    return np.asarray(info["values"], dtype=float)


def _maybe_visualize(state: GameState, title: str, save_path: Path | None, show: bool) -> None:
    if not (NX_AVAILABLE and MATPLOTLIB_AVAILABLE):
        print("Visualization dependencies (networkx/matplotlib) are not installed; skipping plot.")
        return

    import matplotlib
    import matplotlib.pyplot as plt

    if save_path and save_path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
        matplotlib.use("Agg", force=True)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with patch("matplotlib.pyplot.show"):
            visualize_state_mesh(state, title=title)
            plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
    elif show:
        visualize_state_mesh(state, title=title)
    else:
        print("Visualization disabled; nothing to display or save.")


def _format_values(values: Iterable[float]) -> str:
    return "\n".join(
        f"  {power.name:7s}: {prob:.4f}" for power, prob in zip(_STANDARD_POWERS, values)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to an SL checkpoint (.npz)")
    parser.add_argument("--seed", type=int, default=123, help="Seed for random state generation")
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed for random network parameters if needed")
    parser.add_argument("--save", type=Path, default=None, help="Path to save the visualization (png/pdf/svg/etc.)")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive matplotlib display (useful when only saving a file)",
    )
    args = parser.parse_args(argv)

    state = _random_standard_state(seed=args.seed)
    policy = _load_policy(args.checkpoint, rng_seed=args.rng_seed)
    values = _evaluate_values(state, policy)

    total = float(values.sum())
    print("Per-power value probabilities:")
    print(_format_values(values))
    print(f"Sum: {total:.4f}")

    title = f"Random standard-board state (seed={args.seed})"
    _maybe_visualize(state, title, args.save, not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
