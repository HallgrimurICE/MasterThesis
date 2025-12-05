from __future__ import annotations

from typing import Iterable, Optional

from .contracts import Contract
from ..types import Order, OrderType, Power
from ..deepmind.actions import decode_action_to_order


def _is_attack_on(order: Optional[Order], target_power: Power, state) -> bool:
    """Return True if ``order`` directly attacks a province held by ``target_power``."""
    if order is None:
        return False
    if order.type == OrderType.MOVE and order.target:
        unit = state.units.get(order.target)
        if unit is not None and unit.power == target_power:
            return True
        controller = state.supply_center_control.get(order.target)
        if controller == target_power:
            return True
    if order.type == OrderType.SUPPORT and order.support_target:
        unit = state.units.get(order.support_target)
        if unit is not None and unit.power == target_power:
            return True
        controller = state.supply_center_control.get(order.support_target)
        if controller == target_power:
            return True
    return False


def _filter_non_hostile_actions(state, actor: Power, partner: Power, legal_actions: Iterable[int]) -> Iterable[int]:
    allowed = []
    for encoded in legal_actions:
        order = decode_action_to_order(state, actor, int(encoded))
        if not _is_attack_on(order, partner, state):
            allowed.append(int(encoded))
    # Fallback to full legal set if filter removes everything to avoid empty action lists.
    return allowed or list(int(a) for a in legal_actions)


def build_peace_contract(
    state,
    player_i: Power,
    player_j: Power,
    legal_i,
    legal_j,
) -> Contract:
    """Peace contract that filters out actions hostile to the partner."""

    allowed_i = _filter_non_hostile_actions(state, player_i, player_j, legal_i)
    allowed_j = _filter_non_hostile_actions(state, player_j, player_i, legal_j)

    return Contract(
        player_i=player_i,
        player_j=player_j,
        allowed_i=frozenset(allowed_i),
        allowed_j=frozenset(allowed_j),
    )


__all__ = ["build_peace_contract"]
