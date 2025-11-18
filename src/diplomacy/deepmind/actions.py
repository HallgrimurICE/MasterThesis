from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..state import GameState
from ..types import Order, OrderType, Power

from .observation import POSSIBLE_ACTIONS, province_order

from policytraining.environment import action_utils


_PROVINCE_NAME_TO_ID = province_order.province_name_to_id(
    province_order.MapMDF.STANDARD_MAP
)

# The DeepMind map uses a few different abbreviations from the engine board
# (e.g. English Channel is ``ECH`` instead of ``ENG``).  Provide aliases so we
# can translate engine locations into the ids used by the SL observation space.
_PROVINCE_ALIASES = {
    "ENG": "ECH",
    "LYO": "GOL",
    "BOT": "GOB",
}


def _normalise_name(name: str) -> str:
    return name.upper().strip()


def _province_id(name: str) -> int:
    normalised = _normalise_name(name)
    lookup = _PROVINCE_ALIASES.get(normalised, normalised)
    try:
        return _PROVINCE_NAME_TO_ID[lookup]
    except KeyError as exc:  # pragma: no cover - sanity guard
        raise KeyError(f"Unknown province name '{name}'") from exc


def _ordered_powers(state: GameState) -> List[Power]:
    """Define a consistent ordering of powers (players)."""
    return sorted(state.powers, key=str)


def _build_action_tables() -> Tuple[
    Dict[int, int],
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int], int],
    Dict[int, int],
]:
    """Index DeepMind's master action list for quick lookups.

    Returns dictionaries mapping the (order, relevant provinces) combinations we
    need into the encoded 64-bit action values expected by the SL network.
    """

    hold_actions: Dict[int, int] = {}
    move_actions: Dict[Tuple[int, int], int] = {}
    retreat_actions: Dict[Tuple[int, int], int] = {}
    disband_actions: Dict[int, int] = {}

    for encoded in POSSIBLE_ACTIONS.astype(np.int64, copy=False):
        order, src, target, _third = action_utils.action_breakdown(int(encoded))
        src_id = int(src[0])
        if order == action_utils.HOLD:
            hold_actions.setdefault(src_id, int(encoded))
        elif order == action_utils.MOVE_TO:
            tgt_id = int(target[0])
            move_actions.setdefault((src_id, tgt_id), int(encoded))
        elif order == action_utils.RETREAT_TO:
            tgt_id = int(target[0])
            retreat_actions.setdefault((src_id, tgt_id), int(encoded))
        elif order == action_utils.DISBAND:
            disband_actions.setdefault(src_id, int(encoded))

    return hold_actions, move_actions, retreat_actions, disband_actions


_HOLD_ACTIONS, _MOVE_ACTIONS, _RETREAT_ACTIONS, _DISBAND_ACTIONS = (
    _build_action_tables()
)


def _movement_action_encodings(state: GameState, *, power: Power) -> List[int]:
    actions: List[int] = []
    for unit in state.units.values():
        if unit.power != power:
            continue
        src_id = _province_id(unit.loc)
        hold_action = _HOLD_ACTIONS.get(src_id)
        if hold_action is not None:
            actions.append(hold_action)
        for destination in state.legal_moves_from(unit.loc):
            tgt_id = _province_id(destination)
            move_action = _MOVE_ACTIONS.get((src_id, tgt_id))
            if move_action is not None:
                actions.append(move_action)
    return actions


def _retreat_action_encodings(state: GameState, *, power: Power) -> List[int]:
    actions: List[int] = []
    for province, unit in state.pending_retreats.items():
        if unit.power != power:
            continue
        src_id = _province_id(province)
        for destination in state.legal_retreats_from(province):
            tgt_id = _province_id(destination)
            retreat_action = _RETREAT_ACTIONS.get((src_id, tgt_id))
            if retreat_action is not None:
                actions.append(retreat_action)
        disband_action = _DISBAND_ACTIONS.get(src_id)
        if disband_action is not None:
            actions.append(disband_action)
    return actions


def legal_actions_from_state(state: GameState) -> Sequence[np.ndarray]:
    """
    Return legal action encodings in DeepMind's action space for each player.

    First milestone: only needed so the network can pick *some* actions.

    Returns:
        list of np.ndarray, length = num_players.
        Each array contains the encoded actions (64-bit ints) drawn from
        ``policytraining.environment.action_list.POSSIBLE_ACTIONS`` that are
        legal for that player in the current phase.
    """

    powers = _ordered_powers(state)
    legal_actions: List[np.ndarray] = []

    is_retreat_phase = state.phase.name.endswith("RETREAT")

    for power in powers:
        if is_retreat_phase:
            encodings = _retreat_action_encodings(state, power=power)
        else:
            encodings = _movement_action_encodings(state, power=power)
        legal_actions.append(np.asarray(encodings, dtype=np.int64))

    return legal_actions


def decode_actions_to_orders(
    state: GameState,
    power: Power,
    action_indices: Sequence[int],
) -> List[Order]:
    """Convert policy-chosen action indices into engine Order objects.

    The full DeepMind action decoding logic is quite involved.  For now we
    conservatively map every action to a HOLD order for each of the power's
    units so that callers receive a syntactically valid order list.
    """

    del action_indices  # Currently unused in the stub implementation.
    orders: List[Order] = []
    for unit in state.units.values():
        if unit.power != power:
            continue
        orders.append(Order(unit=unit, type=OrderType.HOLD))
    return orders