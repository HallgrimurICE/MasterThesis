from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from policytraining.environment import action_utils

from ..deepmind.actions import _decode_action, _engine_province_name, legal_actions_from_state
from ..state import GameState
from ..types import Order, OrderType, Power, Unit
from .base import Agent


class HeuristicAgent(Agent):
    """Fast baseline agent that scores legal orders with simple heuristics."""

    def __init__(
        self,
        power: Power,
        *,
        rng_seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(power)
        if rng is not None and rng_seed is not None:
            raise ValueError("Provide either rng_seed or rng, not both.")
        self._rng = rng or random.Random(rng_seed)

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        return self._select_orders(state, is_retreat=False)

    def _plan_retreat_orders(self, state: GameState, retreat_index: int) -> List[Order]:
        del retreat_index
        return self._select_orders(state, is_retreat=True)

    def _select_orders(self, state: GameState, *, is_retreat: bool) -> List[Order]:
        legal_actions = self._legal_actions_for_power(state)
        if not legal_actions:
            return []

        candidates_by_unit = self._decode_actions(
            state, legal_actions, is_retreat=is_retreat
        )
        if not candidates_by_unit:
            return []

        high_value = self._high_value_provinces(state)
        enemy_centers = self._enemy_supply_centers(state)
        targets = enemy_centers or high_value

        orders: List[Order] = []
        for loc, candidates in candidates_by_unit.items():
            if not candidates:
                if not is_retreat:
                    unit = state.units.get(loc)
                    if unit is not None:
                        orders.append(Order(unit=unit, type=OrderType.HOLD))
                continue
            safe_move_exists = any(
                order.type == OrderType.MOVE
                and self._enemy_adjacency_count(state, order.target) < 2
                for order in candidates
                if order.target is not None
            )
            scored: List[Tuple[float, Order]] = []
            for order in candidates:
                score = self._score_order(
                    state,
                    order,
                    safe_move_exists=safe_move_exists,
                    high_value=high_value,
                    targets=targets,
                )
                scored.append((score, order))
            best_score = max(score for score, _ in scored)
            best_orders = [order for score, order in scored if score == best_score]
            orders.append(self._rng.choice(best_orders))

        return orders

    def _legal_actions_for_power(self, state: GameState) -> Sequence[int]:
        powers = sorted(state.powers, key=str)
        if self.power not in powers:
            return []
        power_index = powers.index(self.power)
        legal_actions = legal_actions_from_state(state)
        if power_index >= len(legal_actions):
            return []
        return list(legal_actions[power_index])

    def _decode_actions(
        self,
        state: GameState,
        legal_actions: Sequence[int],
        *,
        is_retreat: bool,
    ) -> Dict[str, List[Order]]:
        candidates_by_unit: Dict[str, List[Order]] = {}
        for action in legal_actions:
            order = (
                self._decode_retreat_action(state, action)
                if is_retreat
                else _decode_action(state, self.power, int(action))
            )
            if order is None:
                continue
            candidates_by_unit.setdefault(order.unit.loc, []).append(order)

        if is_retreat:
            for origin, unit in state.pending_retreats.items():
                if unit.power != self.power:
                    continue
                candidates_by_unit.setdefault(origin, [])
        else:
            for unit in state.units.values():
                if unit.power != self.power:
                    continue
                candidates_by_unit.setdefault(unit.loc, [])

        return candidates_by_unit

    def _decode_retreat_action(
        self, state: GameState, action: int
    ) -> Optional[Order]:
        order_code, src, target, _ = action_utils.action_breakdown(int(action))
        src_name = _engine_province_name(int(src[0]))
        pending_unit = state.pending_retreats.get(src_name)
        if pending_unit is None or pending_unit.power != self.power:
            return None
        unit = Unit(self.power, src_name, pending_unit.unit_type)
        if order_code == action_utils.RETREAT_TO:
            target_name = _engine_province_name(int(target[0]))
            return Order(unit=unit, type=OrderType.RETREAT, target=target_name)
        if order_code == action_utils.DISBAND:
            return Order(unit=unit, type=OrderType.RETREAT)
        return None

    def _score_order(
        self,
        state: GameState,
        order: Order,
        *,
        safe_move_exists: bool,
        high_value: Set[str],
        targets: Set[str],
    ) -> float:
        if order.type == OrderType.MOVE and order.target:
            return self._score_move(
                state, order.unit.loc, order.target, high_value, targets
            )
        if order.type == OrderType.RETREAT and order.target:
            return self._score_retreat(state, order.target)
        if order.type == OrderType.HOLD:
            return 1.0 if not safe_move_exists else 0.0
        if order.type == OrderType.RETREAT:
            return -2.0
        if order.type == OrderType.SUPPORT:
            return -0.5
        return 0.0

    def _score_move(
        self,
        state: GameState,
        origin: str,
        destination: str,
        high_value: Set[str],
        targets: Set[str],
    ) -> float:
        score = 0.0
        if destination in state.contested_provinces:
            score += 2.0
        if destination in high_value:
            score += 1.5
        if (
            state.board.get(destination) is not None
            and state.board[destination].is_supply_center
            and state.supply_center_control.get(destination) is None
        ):
            score += 2.0

        if targets:
            origin_dist = self._distance_to_targets(state, origin, targets)
            dest_dist = self._distance_to_targets(state, destination, targets)
            if origin_dist is not None and dest_dist is not None:
                if dest_dist < origin_dist:
                    score += 2.0

        enemy_adjacent = self._enemy_adjacency_count(state, destination)
        if enemy_adjacent >= 2:
            score -= 3.0

        return score

    def _score_retreat(self, state: GameState, destination: str) -> float:
        enemy_adjacent = self._enemy_adjacency_count(state, destination)
        if enemy_adjacent >= 2:
            return -2.0
        return 0.5

    def _enemy_adjacency_count(self, state: GameState, province: str) -> int:
        if province is None:
            return 0
        return sum(
            1
            for neighbor in state.graph.neighbors(province)
            if neighbor in state.units and state.units[neighbor].power != self.power
        )

    def _enemy_supply_centers(self, state: GameState) -> Set[str]:
        centers: Set[str] = set()
        for name, controller in state.supply_center_control.items():
            if controller is None or controller == self.power:
                continue
            centers.add(name)
        return centers

    def _high_value_provinces(self, state: GameState) -> Set[str]:
        if any(prov.is_supply_center for prov in state.board.values()):
            return set()
        return set(sorted(state.board.keys())[:2])

    def _distance_to_targets(
        self, state: GameState, origin: str, targets: Set[str]
    ) -> Optional[int]:
        if origin in targets:
            return 0
        visited = {origin}
        queue = deque([(origin, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in state.graph.neighbors(current):
                if neighbor in visited:
                    continue
                if neighbor in targets:
                    return dist + 1
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
        return None


__all__ = ["HeuristicAgent"]
