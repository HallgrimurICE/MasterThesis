from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence

from ..deepmind.actions import decode_action_to_order, legal_actions_from_state
from ..state import GameState
from ..types import Order, OrderType, Power
from .base import Agent


class HeuristicAgent(Agent):
    """Fast baseline agent that picks legal actions via simple heuristics."""

    def __init__(
        self,
        power: Power,
        *,
        rng_seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        high_value_provinces: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(power)
        if rng is not None:
            self._rng = rng
        else:
            self._rng = random.Random(rng_seed)
        self._high_value_provinces = list(high_value_provinces or [])

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        return self._select_orders(state)

    def _plan_retreat_orders(self, state: GameState, retreat_index: int) -> List[Order]:
        del retreat_index
        return self._select_orders(state)

    def _select_orders(self, state: GameState) -> List[Order]:
        legal_actions = list(legal_actions_from_state(state))
        if not legal_actions:
            return []

        powers = sorted(state.powers, key=str)
        try:
            my_index = powers.index(self.power)
        except ValueError:
            return []

        candidates: Dict[str, List[Order]] = {}
        for encoded in legal_actions[my_index]:
            order = decode_action_to_order(state, self.power, int(encoded))
            if order is None:
                continue
            candidates.setdefault(order.unit.loc, []).append(order)

        targets = self._target_provinces(state)
        orders: List[Order] = []
        for unit_loc, unit in state.units.items():
            if unit.power != self.power:
                continue
            unit_candidates = candidates.get(unit_loc, [])
            if not unit_candidates:
                orders.append(Order(unit=unit, type=OrderType.HOLD))
                continue

            current_distance = self._min_distance(state, unit_loc, targets)
            safe_moves_exist = any(
                self._enemy_adjacent_count(state, order.target) < 2
                for order in unit_candidates
                if order.type == OrderType.MOVE and order.target
            )

            scored = [
                (
                    self._score_order(
                        state,
                        order,
                        targets,
                        current_distance,
                        safe_moves_exist,
                    ),
                    order,
                )
                for order in unit_candidates
            ]
            best_score = max(score for score, _ in scored)
            best_orders = [order for score, order in scored if score == best_score]
            orders.append(self._rng.choice(best_orders))

        return orders

    def _score_order(
        self,
        state: GameState,
        order: Order,
        targets: Sequence[str],
        current_distance: Optional[int],
        safe_moves_exist: bool,
    ) -> float:
        if order.type == OrderType.MOVE and order.target:
            destination = order.target
            score = 0.0
            if destination in state.contested_provinces:
                score += 2.0
            if destination not in state.units:
                province = state.board.get(destination)
                if province and (
                    not province.is_supply_center
                    or state.supply_center_control.get(destination) is None
                ):
                    score += 1.0
            destination_distance = self._min_distance(state, destination, targets)
            if (
                destination_distance is not None
                and current_distance is not None
                and destination_distance < current_distance
            ):
                score += 1.5
            if destination in targets:
                score += 2.0
            if self._enemy_adjacent_count(state, destination) >= 2:
                score -= 2.5
            return score

        if order.type == OrderType.HOLD:
            return 1.0 if not safe_moves_exist else 0.0

        if order.type == OrderType.SUPPORT:
            return 0.0

        if order.type == OrderType.RETREAT:
            if order.target is None:
                return -5.0
            if self._enemy_adjacent_count(state, order.target) >= 2:
                return -2.5
            return 0.0

        return 0.0

    def _enemy_adjacent_count(self, state: GameState, province: str) -> int:
        if not province:
            return 0
        count = 0
        for neighbor in state.graph.neighbors(province):
            unit = state.units.get(neighbor)
            if unit is not None and unit.power != self.power:
                count += 1
        return count

    def _target_provinces(self, state: GameState) -> List[str]:
        supply_centers = [
            name for name, province in state.board.items() if province.is_supply_center
        ]
        if supply_centers:
            return [
                name
                for name in supply_centers
                if state.supply_center_control.get(name) != self.power
            ]

        high_value = list(self._high_value_provinces)
        if not high_value:
            high_value = self._default_high_value_provinces(state)
        return [
            name
            for name in high_value
            if state.units.get(name) is None
            or state.units[name].power != self.power
        ]

    def _default_high_value_provinces(self, state: GameState) -> List[str]:
        ranked = sorted(
            state.board.keys(),
            key=lambda name: (-len(list(state.graph.neighbors(name))), name),
        )
        return ranked[:2]

    @staticmethod
    def _min_distance(
        state: GameState,
        source: str,
        targets: Iterable[str],
    ) -> Optional[int]:
        target_set = set(targets)
        if not target_set:
            return None
        visited = {source}
        queue = deque([(source, 0)])
        while queue:
            current, distance = queue.popleft()
            if current in target_set:
                return distance
            for neighbor in state.graph.neighbors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None


__all__ = ["HeuristicAgent"]
