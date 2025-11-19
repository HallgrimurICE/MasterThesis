"""Helpers for constructing RSS Peace contracts."""

from __future__ import annotations

from typing import List, Sequence

from ..deepmind.actions import decode_action_to_order
from ..state import GameState
from ..types import Order, OrderType, Power
from .contracts import Contract


def _attacks_power(order: Order | None, target_power: Power, state: GameState) -> bool:
    """Return True if ``order`` attacks ``target_power``'s units or SCs."""

    if order is None:
        return False
    if order.type not in {OrderType.MOVE, OrderType.SUPPORT, OrderType.RETREAT}:
        return False

    def threatens_location(location: str | None) -> bool:
        if not location:
            return False
        occupying_unit = state.units.get(location)
        if occupying_unit and occupying_unit.power == target_power:
            return True
        controller = state.supply_center_control.get(location)
        if controller == target_power:
            return True
        province = state.board.get(location)
        return bool(province and province.is_supply_center and province.home_power == target_power)

    if order.type == OrderType.SUPPORT:
        return threatens_location(order.support_target)
    return threatens_location(order.target)


def _filter_non_aggressive_actions(
    state: GameState,
    acting_power: Power,
    target_power: Power,
    legal_actions: Sequence[int],
) -> List[int]:
    allowed: List[int] = []
    for encoded in legal_actions:
        order = decode_action_to_order(state, acting_power, int(encoded))
        if not _attacks_power(order, target_power, state):
            allowed.append(int(encoded))
    return allowed


def build_peace_contract(
    state: GameState,
    power_i: Power,
    power_j: Power,
    legal_i: Sequence[int],
    legal_j: Sequence[int],
) -> Contract:
    """Return a Peace contract restricting ``power_i`` and ``power_j``.

    Peace contracts forbid actions that attack the other party's units or supply
    centers.  We follow the Round-Simultaneous Signaling (RSS) definition from
    the DeepMind negotiation agent and encode the safe action sets as
    ``Contract`` objects.
    """

    safe_i = frozenset(_filter_non_aggressive_actions(state, power_i, power_j, legal_i))
    safe_j = frozenset(_filter_non_aggressive_actions(state, power_j, power_i, legal_j))
    return Contract(power_i=power_i, power_j=power_j, allowed_i=safe_i, allowed_j=safe_j)


__all__ = ["build_peace_contract"]
