# test_sl_on_state.py

from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    script_path = os.path.realpath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    src_dir = os.path.join(project_root, "src")
    policytraining_dir = os.path.join(project_root, "policytraining")

    for path in (src_dir, policytraining_dir, project_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    script_dir = os.path.dirname(script_path)
    while script_dir in sys.path:
        sys.path.remove(script_dir)

import argparse
from pathlib import Path

from .maps import standard_board
from .state import GameState
from .types import Power, Unit, UnitType

from .deepmind import build_observation, legal_actions_from_state
from policytraining.run_sl import make_sl_policy


def main(sl_params_path: Path) -> None:
    # 1) Build a simple test GameState
    board = standard_board()
    units = {
        "PAR": Unit(Power("France"), "PAR", UnitType.ARMY),
        "ION": Unit(Power("Italy"), "ION", UnitType.FLEET),
    }
    powers = {Power("France"), Power("Italy")}
    state = GameState(board=board, units=units, powers=powers)

    # 2) DeepMind observation + legal actions
    obs = build_observation(state, last_actions=[42])
    legal_actions = legal_actions_from_state(state)

    # 3) Load SL policy
    policy = make_sl_policy(str(sl_params_path))

    # 4) Ask for actions for all players
    slots_list = list(range(len(legal_actions)))
    actions, info = policy.actions(
        slots_list=slots_list,
        observation=obs,
        legal_actions=legal_actions,
    )

    print("Slots list:", slots_list)
    print("Chosen action indices per slot:", actions)
    print("Values:", info["values"])
    print("Policy shape:", info["policy"].shape)


def _default_params_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "policytraining" / "data" / "sl_params.npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for the SL policy on a toy state.")
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Path to the NPZ file containing the supervised learning parameters.",
    )
    args = parser.parse_args()

    sl_params_path = args.params or _default_params_path()
    if not sl_params_path.is_file():
        raise FileNotFoundError(
            f"Could not find supervised learning parameters at {sl_params_path}. "
            "Use --params to point to the correct NPZ file."
        )

    main(sl_params_path)
