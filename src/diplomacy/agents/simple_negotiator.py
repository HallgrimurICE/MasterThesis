from __future__ import annotations

from collections import OrderedDict
import random
from typing import Iterable, List, Set, Tuple

from .base import Agent
from .best_response import SampledBestResponsePolicy
from ..state import GameState
from ..types import Order, OrderType, Power, UnitType


def _is_attack_on(order: Order, target_power: Power, state: GameState) -> bool:
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


def _is_supporting_partner(order: Order, partner: Power, state: GameState) -> bool:
    if order.type != OrderType.SUPPORT:
        return False
    if order.support_target:
        unit = state.units.get(order.support_target)
        return unit is not None and unit.power == partner
    if order.support_unit_loc:
        unit = state.units.get(order.support_unit_loc)
        return unit is not None and unit.power == partner
    return False


def _restrict_candidate_map(
    state: GameState,
    candidate_map: "OrderedDict[str, List[Order]]",
    peace_partners: Set[Power],
) -> "OrderedDict[str, List[Order]]":
    if not peace_partners:
        return candidate_map
    restricted: "OrderedDict[str, List[Order]]" = OrderedDict()
    for loc, orders in candidate_map.items():
        allowed = [
            order
            for order in orders
            if not any(_is_attack_on(order, partner, state) for partner in peace_partners)
        ]
        restricted[loc] = allowed or list(orders)
    return restricted


def _support_candidate_maps(
    state: GameState,
    candidate_map: "OrderedDict[str, List[Order]]",
    partner: Power,
) -> List["OrderedDict[str, List[Order]]"]:
    supporting_units = []
    for loc, orders in candidate_map.items():
        if any(_is_supporting_partner(order, partner, state) for order in orders):
            supporting_units.append(loc)

    if not supporting_units:
        return []

    variants: List["OrderedDict[str, List[Order]]"] = []
    for loc in supporting_units:
        restricted = OrderedDict()
        for unit_loc, orders in candidate_map.items():
            if unit_loc == loc:
                support_orders = [
                    order for order in orders if _is_supporting_partner(order, partner, state)
                ]
                restricted[unit_loc] = support_orders or list(orders)
            else:
                restricted[unit_loc] = list(orders)
        variants.append(restricted)
    return variants


class SimpleNegotiatorAgent(Agent):
    """Negotiator that uses a lightweight order policy and peace filtering."""

    def __init__(
        self,
        power: Power,
        *,
        policy: SampledBestResponsePolicy | None = None,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(power)
        self._policy = policy or SampledBestResponsePolicy(rng=random.Random(rng_seed))
        self._peace_partners: Set[Power] = set()
        self._support_partners: Set[Power] = set()

    def set_peace_partners(self, partners: Iterable[Power]) -> None:
        self._peace_partners = set(partners)

    def set_support_partners(self, partners: Iterable[Power]) -> None:
        self._support_partners = set(partners)

    def plan_orders_with_peace(
        self,
        state: GameState,
        partners: Iterable[Power],
    ) -> List[Order]:
        candidate_map = self._policy._build_candidate_order_map(state, self.power)  # type: ignore[attr-defined]
        restricted = _restrict_candidate_map(state, candidate_map, set(partners))
        best_orders, _ = self._policy._select_best_orders(  # type: ignore[attr-defined]
            state,
            self.power,
            restricted,
        )
        return list(best_orders)

    def plan_orders_with_support(
        self,
        state: GameState,
        partner: Power,
        *,
        peace_partners: Iterable[Power] = (),
    ) -> Tuple[List[Order], float]:
        candidate_map = self._policy._build_candidate_order_map(state, self.power)  # type: ignore[attr-defined]
        restricted = _restrict_candidate_map(state, candidate_map, set(peace_partners))
        variants = _support_candidate_maps(state, restricted, partner)
        if not variants:
            orders, score = self._policy._select_best_orders(  # type: ignore[attr-defined]
                state,
                self.power,
                restricted,
            )
            return list(orders), score

        best_orders: List[Order] = []
        best_score = float("-inf")
        for variant in variants:
            orders, score = self._policy._select_best_orders(  # type: ignore[attr-defined]
                state,
                self.power,
                variant,
            )
            if score > best_score:
                best_score = score
                best_orders = list(orders)
        return best_orders, best_score

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        if not self._support_partners:
            return self.plan_orders_with_peace(state, self._peace_partners)

        peace_partners = set(self._peace_partners) | set(self._support_partners)
        best_orders: List[Order] = []
        best_score = float("-inf")
        for partner in sorted(self._support_partners, key=str):
            orders, score = self.plan_orders_with_support(
                state,
                partner,
                peace_partners=peace_partners,
            )
            if score > best_score:
                best_score = score
                best_orders = orders

        if best_orders:
            return best_orders
        return self.plan_orders_with_peace(state, peace_partners)

    def plan_builds(
        self,
        state: GameState,
        build_count: int,
    ) -> List[Tuple[str, UnitType]]:
        return self._policy.plan_builds(state, self.power, build_count)


__all__ = ["SimpleNegotiatorAgent"]
