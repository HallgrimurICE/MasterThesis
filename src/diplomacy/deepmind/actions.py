from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..state import GameState
from ..types import Order, OrderType, Power

from .observation import POSSIBLE_ACTIONS, province_order

from policytraining.environment import action_utils


_PROVINCE_NAME_TO_ID = province_order.province_name_to_id(
    province_order.MapMDF.STANDARD_MAP
)
_PROVINCE_ID_TO_NAME = {
    province_id: name for name, province_id in _PROVINCE_NAME_TO_ID.items()
}

# The DeepMind map uses a few different abbreviations from the engine board
# (e.g. English Channel is ``ECH`` instead of ``ENG``).  Provide aliases so we
# can translate engine locations into the ids used by the SL observation space.
_PROVINCE_ALIASES = {
    "ENG": "ECH",
    "LYO": "GOL",
    "BOT": "GOB",
}

_ENGINE_NAME_OVERRIDES = {deep: eng for eng, deep in _PROVINCE_ALIASES.items()}


def _normalise_name(name: str) -> str:
    return name.upper().strip()


def _province_id(name: str) -> int:
    normalised = _normalise_name(name)
    lookup = _PROVINCE_ALIASES.get(normalised, normalised)
    try:
        return _PROVINCE_NAME_TO_ID[lookup]
    except KeyError as exc:  # pragma: no cover - sanity guard
        raise KeyError(f"Unknown province name '{name}'") from exc


def _engine_province_name(province_id: int) -> str:
    """Map a DeepMind province id back to the engine's canonical name."""

    deep_name = _PROVINCE_ID_TO_NAME[province_id]
    base = deep_name.split("_")[0]
    return _ENGINE_NAME_OVERRIDES.get(base, base)


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

    The DeepMind policy emits encoded actions drawn from the master catalogue in
    ``policytraining.environment.action_list``.  Decode those 64-bit integers
    back into board-relative orders so that the adjudicator can process them.
    """
    orders_by_loc: Dict[str, Order] = {}

    for encoded in action_indices:
        order = _decode_action(state, power, int(encoded))
        if order is None:
            continue
        loc = order.unit.loc
        if loc in orders_by_loc:
            continue
        orders_by_loc[loc] = order

    # Ensure every unit receives at least a hold so the adjudicator always sees
    # a complete order set.
    for unit in state.units.values():
        if unit.power != power:
            continue
        orders_by_loc.setdefault(unit.loc, Order(unit=unit, type=OrderType.HOLD))

    return list(orders_by_loc.values())


def _decode_action(state: GameState, power: Power, encoded: int) -> Optional[Order]:
    order_code, src, target, third = action_utils.action_breakdown(encoded)
    src_name = _engine_province_name(int(src[0]))
    unit = state.units.get(src_name)
    if unit is None or unit.power != power:
        return None

    if order_code == action_utils.HOLD:
        return Order(unit=unit, type=OrderType.HOLD)

    if order_code == action_utils.MOVE_TO:
        target_name = _engine_province_name(int(target[0]))
        return Order(unit=unit, type=OrderType.MOVE, target=target_name)

    if order_code == action_utils.SUPPORT_HOLD:
        support_loc = _engine_province_name(int(target[0]))
        return Order(
            unit=unit,
            type=OrderType.SUPPORT,
            support_unit_loc=support_loc,
        )

    if order_code == action_utils.SUPPORT_MOVE_TO:
        support_loc = _engine_province_name(int(third[0]))
        support_target = _engine_province_name(int(target[0]))
        return Order(
            unit=unit,
            type=OrderType.SUPPORT,
            support_unit_loc=support_loc,
            support_target=support_target,
        )

    if order_code == action_utils.RETREAT_TO:
        target_name = _engine_province_name(int(target[0]))
        return Order(unit=unit, type=OrderType.RETREAT, target=target_name)

    if order_code == action_utils.DISBAND:
        return Order(unit=unit, type=OrderType.RETREAT)

    # Convoys, builds, and removes are not modelled in the lightweight engine
    # yet, so fall back to a hold for unsupported action codes.
    return Order(unit=unit, type=OrderType.HOLD)


def decode_action_to_order(state: GameState, power: Power, encoded: int) -> Optional[Order]:
    """Public helper to decode a single encoded action into an ``Order``.

    This thin wrapper exposes ``_decode_action`` so other modules (e.g. the RSS
    negotiation helpers) can reason about whether a given move would attack a
    particular power.
    """

    return _decode_action(state, power, encoded)