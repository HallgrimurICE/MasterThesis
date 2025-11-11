from __future__ import annotations

import itertools
import math
import random
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple

from ..adjudication import Adjudicator
from ..orders import hold, move, retreat, support_hold, support_move
from ..state import GameState
from ..types import Order, Unit
from .base import Agent


class BaselineNegotiator(Agent):
    """Simple SBR-inspired negotiator that evaluates sampled order sets.

    The implementation is intentionally lightweight – it does not attempt to
    reproduce DeepMind's full negotiation stack, but it mirrors the high-level
    idea of Sampled Best Response (SBR): generate candidate joint orders for a
    power, roll them out against sampled opponent behaviour, and pick the set
    with the highest expected value.  The evaluator favours securing supply
    centres, keeping friendly units alive, and avoiding threatened positions.

    Parameters
    ----------
    power:
        The power controlled by this agent.
    max_candidates:
        Maximum number of joint order combinations to evaluate per round.  When
        the Cartesian product of unit order options exceeds this bound we fall
        back to random sampling while ensuring a deterministic baseline (all
        holds) is evaluated.
    opponent_samples:
        Number of opponent order samples used when scoring a joint order set.
    rng:
        Optional ``random.Random`` instance for deterministic behaviour.
    """

    def __init__(
        self,
        power,
        *,
        max_candidates: int = 128,
        opponent_samples: int = 3,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(power)
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if opponent_samples <= 0:
            raise ValueError("opponent_samples must be positive")
        self.max_candidates = max_candidates
        self.opponent_samples = opponent_samples
        self._rng = rng or random.Random()

    def _plan_orders(self, state: "GameState", round_index: int) -> List[Order]:
        friendly_units = [
            unit for unit in state.units.values() if unit.power == self.power
        ]
        if not friendly_units:
            return []

        per_unit_orders: List[List[Order]] = [
            self._candidate_orders_for_unit(unit, state) for unit in friendly_units
        ]

        joint_orders = self._sample_joint_orders(per_unit_orders)
        baseline_orders = [hold(unit) for unit in friendly_units]
        best_orders: Sequence[Order] = baseline_orders
        best_score = float("-inf")

        for combo in joint_orders:
            score = self._evaluate_orders(state, combo)
            if score > best_score:
                best_score = score
                best_orders = combo
            elif math.isclose(score, best_score):
                # Tie-break deterministically using lexicographic order strings.
                if self._orders_key(combo) < self._orders_key(best_orders):
                    best_orders = combo

        return list(best_orders)

    def _plan_retreat_orders(
        self, state: "GameState", retreat_index: int
    ) -> List[Order]:
        orders: List[Order] = []
        for origin, pending in state.pending_retreats.items():
            if pending.power != self.power:
                continue
            legal = state.legal_retreats_from(origin)
            if not legal:
                continue
            unit = Unit(self.power, origin, pending.unit_type)
            best_dest = max(
                legal,
                key=lambda dest: self._retreat_destination_score(unit, dest, state),
            )
            orders.append(retreat(unit, best_dest))
        return orders

    def _candidate_orders_for_unit(self, unit: Unit, state: "GameState") -> List[Order]:
        options: List[Order] = []

        # Hold is always available.
        options.append(hold(unit))

        # Direct moves into legal neighbouring provinces.
        for destination in state.legal_moves_from(unit.loc):
            options.append(move(unit, destination))

        # Support actions for adjacent friendly units.
        for friend in state.units.values():
            if friend.power != self.power or friend.loc == unit.loc:
                continue
            if friend.loc not in state.graph.neighbors(unit.loc):
                continue
            options.append(support_hold(unit, friend.loc))
            for friend_dest in state.legal_moves_from(friend.loc):
                if friend_dest not in state.graph.neighbors(unit.loc):
                    continue
                options.append(support_move(unit, friend.loc, friend_dest))

        # Remove duplicates while preserving order.
        seen = set()
        deduped: List[Order] = []
        for order in options:
            key = self._order_identity(order)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(order)
        return deduped

    def _sample_joint_orders(
        self, per_unit_orders: Sequence[Sequence[Order]]
    ) -> List[Sequence[Order]]:
        total = math.prod(len(opts) for opts in per_unit_orders)
        if total <= self.max_candidates:
            combos = list(itertools.product(*per_unit_orders))
        else:
            combos_set: set[Tuple[Order, ...]] = set()
            # Ensure the all-hold baseline is evaluated.
            baseline = tuple(opts[0] for opts in per_unit_orders)
            combos_set.add(baseline)
            while len(combos_set) < self.max_candidates:
                selection = tuple(
                    self._rng.choice(list(opts)) for opts in per_unit_orders
                )
                combos_set.add(selection)
            combos = list(combos_set)

        combos.sort(key=self._orders_key)
        return combos

    def _evaluate_orders(
        self, state: "GameState", orders: Sequence[Order]
    ) -> float:
        score = 0.0
        for _ in range(self.opponent_samples):
            opponent_orders = list(self._sample_opponent_orders(state))
            combined: List[Order] = list(orders) + opponent_orders
            next_state, _ = Adjudicator(state).resolve(combined)
            score += self._state_value(next_state)
        return score / float(self.opponent_samples)

    def _sample_opponent_orders(self, state: "GameState") -> Iterable[Order]:
        for unit in state.units.values():
            if unit.power == self.power:
                continue
            legal = state.legal_moves_from(unit.loc)
            if not legal or self._rng.random() < 0.5:
                yield hold(unit)
            else:
                destination = self._rng.choice(legal)
                yield move(unit, destination)

    def _state_value(self, state: "GameState") -> float:
        score = 0.0
        for loc, unit in state.units.items():
            province = state.board.get(loc)
            if province is None:
                continue
            if unit.power == self.power:
                score += 1.0
                if province.is_supply_center:
                    score += 2.0
            else:
                if province.is_supply_center:
                    score -= 1.5

        score += state.supply_centers(self.power) * 1.5
        score -= state.centers_threatened(self.power) * 0.5

        # Penalise outstanding retreats for this power.
        pending = sum(
            1
            for retreat_unit in state.pending_retreats.values()
            if retreat_unit.power == self.power
        )
        score -= pending * 1.0

        # Small randomised jitter for deterministic tie-breaking influenced by rng.
        score += self._rng.random() * 0.01
        return score

    def _retreat_destination_score(
        self, unit: Unit, destination: str, state: "GameState"
    ) -> float:
        province = state.board.get(destination)
        base = 0.0
        if province is not None and province.is_supply_center:
            base += 1.5

        # Prefer destinations further from enemy units.
        enemy_distance = min(
            (
                self._graph_distance(destination, enemy.loc, state)
                for enemy in state.units.values()
                if enemy.power != self.power
            ),
            default=3,
        )
        base += 0.3 * enemy_distance
        return base

    def _graph_distance(self, start: str, goal: str, state: "GameState") -> int:
        if start == goal:
            return 0

        visited = {start}
        queue: deque[Tuple[str, int]] = deque([(start, 0)])
        while queue:
            current, distance = queue.popleft()
            for neighbor in state.graph.neighbors(current):
                if neighbor == goal:
                    return distance + 1
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return 3

    @staticmethod
    def _order_identity(order: Order) -> Tuple:
        return (
            order.unit.power,
            order.unit.loc,
            order.type,
            order.target,
            order.support_unit_loc,
            order.support_target,
        )

    @staticmethod
    def _orders_key(orders: Sequence[Order]) -> Tuple[str, ...]:
        return tuple(str(order) for order in orders)


__all__ = ["BaselineNegotiator"]
