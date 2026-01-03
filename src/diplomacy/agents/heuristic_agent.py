from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ..deepmind.actions import decode_action_to_order, legal_actions_from_state
from ..orders import hold
from ..state import GameState
from ..types import Order, OrderType, Power, Unit
from .base import Agent


class HeuristicAgent(Agent):
    """Fast baseline agent that scores legal moves with simple rules."""

    def __init__(
        self,
        power: Power,
        *,
        rng_seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(power)
        if rng is not None and rng_seed is not None:
            raise ValueError("Provide either rng or rng_seed, not both.")
        if rng is not None:
            self._rng = rng
        else:
            self._rng = random.Random(rng_seed)

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        legal_actions = legal_actions_from_state(state)
        if not legal_actions:
            return []

        powers = sorted(state.powers, key=str)
        try:
            power_index = powers.index(self.power)
        except ValueError:
            return []

        my_actions = legal_actions[power_index]
        candidates_by_loc: Dict[str, List[Order]] = {}

        for encoded in my_actions:
            order = decode_action_to_order(state, self.power, int(encoded))
            if order is None:
                continue
            if order.type not in {OrderType.HOLD, OrderType.MOVE}:
                continue
            candidates_by_loc.setdefault(order.unit.loc, []).append(order)

        orders: List[Order] = []
        for unit in state.units.values():
            if unit.power != self.power:
                continue
            candidates = candidates_by_loc.get(unit.loc, [])
            if not candidates:
                orders.append(hold(unit))
                continue
            orders.append(self._choose_order(state, unit, candidates))

        return orders

    def _plan_retreat_orders(
        self,
        state: GameState,
        retreat_index: int,
    ) -> List[Order]:
        del retreat_index
        legal_actions = legal_actions_from_state(state)
        if not legal_actions:
            return []

        powers = sorted(state.powers, key=str)
        try:
            power_index = powers.index(self.power)
        except ValueError:
            return []

        orders: Dict[str, Order] = {}
        for encoded in legal_actions[power_index]:
            order = decode_action_to_order(state, self.power, int(encoded))
            if order is None or order.type != OrderType.RETREAT:
                continue
            existing = orders.get(order.unit.loc)
            if existing is None:
                orders[order.unit.loc] = order
            elif existing.target is None and order.target is not None:
                orders[order.unit.loc] = order

        return list(orders.values())

    def _choose_order(
        self,
        state: GameState,
        unit: Unit,
        candidates: Sequence[Order],
    ) -> Order:
        hold_candidates = [order for order in candidates if order.type == OrderType.HOLD]
        move_candidates = [
            order for order in candidates if order.type == OrderType.MOVE and order.target
        ]

        safe_moves = [
            order
            for order in move_candidates
            if not self._is_unsafe_destination(state, order.target)
        ]

        if safe_moves:
            scored = [
                (self._score_move(state, unit, order.target), order)
                for order in safe_moves
            ]
            best_score = max(score for score, _ in scored)
            best_orders = [order for score, order in scored if score == best_score]
            if len(best_orders) == 1:
                return best_orders[0]
            return self._rng.choice(best_orders)

        if hold_candidates:
            return hold_candidates[0]
        if move_candidates:
            return move_candidates[0]
        return hold(unit)

    def _score_move(self, state: GameState, unit: Unit, destination: str) -> float:
        score = 0.0

        if destination in state.contested_provinces:
            score += 1.0

        if self._is_neutral_supply_center(state, destination):
            score += 1.0

        target_centers = self._target_centers(state)
        if destination in target_centers:
            score += 2.0

        if self._moves_toward_target(state, unit.loc, destination, target_centers):
            score += 1.0

        if self._is_unsafe_destination(state, destination):
            score -= 2.0

        return score

    def _is_neutral_supply_center(self, state: GameState, province: str) -> bool:
        if not state.board.get(province, None):
            return False
        if not state.board[province].is_supply_center:
            return False
        return state.supply_center_control.get(province) is None

    def _target_centers(self, state: GameState) -> List[str]:
        supply_centers = [
            name
            for name, province in state.board.items()
            if province.is_supply_center
        ]
        if supply_centers:
            return [
                name
                for name in supply_centers
                if state.supply_center_control.get(name) != self.power
            ]
        return sorted(state.board.keys())[:2]

    def _moves_toward_target(
        self,
        state: GameState,
        origin: str,
        destination: str,
        targets: Iterable[str],
    ) -> bool:
        target_list = list(targets)
        if not target_list:
            return False
        origin_distance = self._distance_to_targets(state, origin, target_list)
        dest_distance = self._distance_to_targets(state, destination, target_list)
        if origin_distance is None or dest_distance is None:
            return False
        return dest_distance < origin_distance

    def _distance_to_targets(
        self,
        state: GameState,
        origin: str,
        targets: Iterable[str],
    ) -> Optional[int]:
        target_set = set(targets)
        if origin in target_set:
            return 0
        visited = {origin}
        frontier: Deque[Tuple[str, int]] = deque([(origin, 0)])
        while frontier:
            current, distance = frontier.popleft()
            next_distance = distance + 1
            for neighbor in state.graph.neighbors(current):
                if neighbor in visited:
                    continue
                if neighbor in target_set:
                    return next_distance
                visited.add(neighbor)
                frontier.append((neighbor, next_distance))
        return None

    def _is_unsafe_destination(self, state: GameState, destination: str) -> bool:
        enemy_neighbors = 0
        for neighbor in state.graph.neighbors(destination):
            unit = state.units.get(neighbor)
            if unit is not None and unit.power != self.power:
                enemy_neighbors += 1
        return enemy_neighbors >= 2


__all__ = ["HeuristicAgent"]
