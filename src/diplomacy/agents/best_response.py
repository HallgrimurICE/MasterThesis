from __future__ import annotations

import itertools
import random
from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple

from ..adjudication import Adjudicator
from ..orders import hold, move, support_hold, support_move
from ..state import GameState
from ..types import Order, Power, Unit, UnitType, ProvinceType
from .base import Agent


class SampledBestResponsePolicy:
    """Approximate best-response policy using sampled best responses.

    The policy enumerates or samples joint orders for the controlled power and
    evaluates them against a set of stochastic opponent base profiles,
    mirroring the sampled-best-response procedure. Candidate sets are limited
    for tractability and scored with the linear value weights already tuned for
    the project. When ``rollout_depth`` exceeds one the policy recursively
    re-applies the search to propagate value estimates across successive
    seasons, discounting deeper plies. The same rollout logic is reused when
    choosing between army and fleet builds so coastal decisions consider their
    downstream order quality.
    """

    def __init__(
        self,
        *,
        rollout_limit: int = 64,
        rollout_depth: int = 1,
        rollout_discount: float = 0.9,
        rng: random.Random | None = None,
        unit_weight: float = 1.0,
        supply_center_weight: float = 5.0,
        threatened_penalty: float = 2.0,
        base_profile_count: int = 8,
    ) -> None:
        self.rollout_limit = max(1, rollout_limit)
        self._rng = rng or random.Random()
        self.unit_weight = unit_weight
        self.supply_center_weight = supply_center_weight
        self.threatened_penalty = threatened_penalty
        self.base_profile_count = max(1, base_profile_count)
        self.rollout_depth = max(1, rollout_depth)
        self.rollout_discount = max(0.0, min(1.0, rollout_discount))

    def plan_orders(self, state: GameState, power: Power) -> List[Order]:
        candidate_map = self._build_candidate_order_map(state, power)
        best_orders, _ = self._select_best_orders(
            state, power, candidate_map, depth=self.rollout_depth
        )
        return best_orders

    def _candidate_orders(self, state: GameState, unit: Unit) -> List[Order]:
        moves = state.legal_moves_from(unit.loc)
        orders = [hold(unit)]
        orders.extend(move(unit, destination) for destination in moves)

        friendly_units = [
            other for other in state.units.values() if other.power == unit.power
        ]
        legal_moves_map = {
            other.loc: state.legal_moves_from(other.loc) for other in friendly_units
        }

        support_hold_targets = sorted(
            nbr
            for nbr in state.graph.neighbors(unit.loc)
            if nbr in state.units and state.units[nbr].power == unit.power
        )
        orders.extend(support_hold(unit, friend) for friend in support_hold_targets)

        support_move_options = {
            (friend.loc, destination)
            for destination in state.graph.neighbors(unit.loc)
            for friend in friendly_units
            if friend.loc != unit.loc
            and destination in legal_moves_map.get(friend.loc, [])
        }
        orders.extend(
            support_move(unit, friend_from, friend_to)
            for friend_from, friend_to in sorted(support_move_options)
        )

        return orders

    def plan_builds(
        self,
        state: GameState,
        power: Power,
        build_count: int,
    ) -> List[Tuple[str, UnitType]]:
        if build_count <= 0:
            return []
        candidates = self._build_candidates(state, power)
        if not candidates:
            return []
        combos = self._enumerate_build_combos(candidates, build_count)
        best_choice: Optional[Sequence[Tuple[str, UnitType]]] = None
        best_score = float("-inf")
        for combo in combos:
            test_state = state.copy()
            for loc, unit_type in combo:
                test_state.units[loc] = Unit(power, loc, unit_type)
            candidate_map = self._build_candidate_order_map(test_state, power)
            _, score = self._select_best_orders(
                test_state, power, candidate_map, depth=self.rollout_depth
            )
            if score > best_score:
                best_score = score
                best_choice = combo
        if best_choice is None:
            return []
        return list(best_choice)

    def _build_candidate_order_map(
        self, state: GameState, power: Power
    ) -> "OrderedDict[str, List[Order]]":
        friendly_units = sorted(
            (unit for unit in state.units.values() if unit.power == power),
            key=lambda unit: unit.loc,
        )
        candidate_map: "OrderedDict[str, List[Order]]" = OrderedDict()
        for unit in friendly_units:
            candidate_map[unit.loc] = self._candidate_orders(state, unit)
        return candidate_map

    def _select_best_orders(
        self,
        state: GameState,
        power: Power,
        candidate_map: "OrderedDict[str, List[Order]]" | None = None,
        *,
        depth: int | None = None,
    ) -> Tuple[List[Order], float]:
        if depth is None:
            depth = self.rollout_depth
        depth = max(1, depth)
        if candidate_map is None:
            candidate_map = self._build_candidate_order_map(state, power)

        if not candidate_map:
            base_score = self._evaluate_state(state, None, power)
            return [], base_score

        combos = self._enumerate_combos(candidate_map)
        if not combos:
            base_score = self._evaluate_state(state, None, power)
            return [], base_score

        base_profiles = self._sample_base_profiles(state, power)
        unit_order = list(candidate_map.keys())

        best_score = float("-inf")
        best_orders: Sequence[Order] | None = None
        for combo in combos:
            score = self._estimate_combo_value(
                state, power, unit_order, combo, base_profiles, depth
            )
            if score > best_score:
                best_score = score
                best_orders = combo

        if best_orders is None:
            return [], self._evaluate_state(state, None, power)
        return list(best_orders), best_score

    def _enumerate_combos(
        self,
        candidate_map: "OrderedDict[str, List[Order]]",
    ) -> List[Tuple[Order, ...]]:
        candidate_lists = list(candidate_map.values())
        if not candidate_lists:
            return []

        total = 1
        for options in candidate_lists:
            total *= len(options)
            if total > self.rollout_limit:
                break

        if total <= self.rollout_limit:
            return list(itertools.product(*candidate_lists))

        samples = set()
        # Ensure deterministic baseline (all holds).
        baseline = tuple(options[0] for options in candidate_lists)
        samples.add(baseline)
        while len(samples) < self.rollout_limit:
            selection = tuple(
                self._rng.choice(options) for options in candidate_lists
            )
            samples.add(selection)
        return list(samples)

    def _build_candidates(
        self,
        state: GameState,
        power: Power,
    ) -> List[Tuple[str, Tuple[UnitType, ...]]]:
        candidates: List[Tuple[str, Tuple[UnitType, ...]]] = []
        for loc in sorted(state.available_build_sites(power)):
            province = state.board.get(loc)
            if province is None:
                continue
            if province.province_type == ProvinceType.SEA:
                continue
            if province.province_type == ProvinceType.COAST:
                unit_types = (UnitType.ARMY, UnitType.FLEET)
            else:
                unit_types = (UnitType.ARMY,)
            candidates.append((loc, unit_types))
        return candidates

    def _enumerate_build_combos(
        self,
        candidates: List[Tuple[str, Tuple[UnitType, ...]]],
        build_count: int,
    ) -> List[Tuple[Tuple[str, UnitType], ...]]:
        count = min(build_count, len(candidates))
        if count <= 0:
            return []

        combos: set[Tuple[Tuple[str, UnitType], ...]] = set()
        baseline = tuple(
            (candidates[idx][0], candidates[idx][1][0]) for idx in range(count)
        )
        combos.add(tuple(sorted(baseline)))

        max_attempts = max(self.rollout_limit * 4, 32)
        attempts = 0
        while len(combos) < self.rollout_limit and attempts < max_attempts:
            indices = sorted(self._rng.sample(range(len(candidates)), count))
            placement = []
            for idx in indices:
                loc, unit_types = candidates[idx]
                placement.append((loc, self._rng.choice(unit_types)))
            combos.add(tuple(sorted(placement)))
            attempts += 1

        if len(combos) < self.rollout_limit:
            for subset in itertools.combinations(range(len(candidates)), count):
                choice_lists = [
                    [(candidates[idx][0], unit_type) for unit_type in candidates[idx][1]]
                    for idx in subset
                ]
                for combo in itertools.product(*choice_lists):
                    combos.add(tuple(sorted(combo)))
                    if len(combos) >= self.rollout_limit:
                        break
                if len(combos) >= self.rollout_limit:
                    break

        return [combo for combo in combos]

    def _sample_base_profiles(
        self, state: GameState, power: Power
    ) -> List[Tuple[Order, ...]]:
        opponents = sorted(
            (unit for unit in state.units.values() if unit.power != power),
            key=lambda unit: (str(unit.power), unit.loc),
        )
        if not opponents:
            return [tuple()]

        all_units: List[Unit] = list(state.units.values())
        legal_moves_map = {
            unit.loc: state.legal_moves_from(unit.loc) for unit in all_units
        }

        profiles: List[Tuple[Order, ...]] = []
        baseline = tuple(hold(unit) for unit in opponents)
        profiles.append(baseline)

        while len(profiles) < self.base_profile_count:
            profile_orders = []
            for unit in opponents:
                profile_orders.append(
                    self._sample_opponent_order(state, unit, all_units, legal_moves_map)
                )
            profiles.append(tuple(profile_orders))

        return profiles

    def _sample_opponent_order(
        self,
        state: GameState,
        unit: Unit,
        all_units: Sequence[Unit],
        legal_moves_map: dict[str, List[str]],
    ) -> Order:
        legal_moves = legal_moves_map.get(unit.loc, [])
        support_hold_targets = [
            nbr
            for nbr in state.graph.neighbors(unit.loc)
            if nbr in state.units and state.units[nbr].power == unit.power
        ]
        support_move_options: List[Tuple[str, str]] = []
        for dest in state.graph.neighbors(unit.loc):
            for other_unit in all_units:
                if other_unit.loc == unit.loc:
                    continue
                if other_unit.power != unit.power:
                    continue
                other_moves = legal_moves_map.get(other_unit.loc, [])
                if dest in other_moves:
                    support_move_options.append((other_unit.loc, dest))

        options: List[Order] = [hold(unit)]
        options.extend(move(unit, destination) for destination in legal_moves)
        options.extend(support_hold(unit, friend) for friend in support_hold_targets)
        options.extend(
            support_move(unit, friend_from, friend_to)
            for friend_from, friend_to in support_move_options
        )

        return self._rng.choice(options)

    def _estimate_combo_value(
        self,
        state: GameState,
        power: Power,
        unit_order: Sequence[str],
        combo: Sequence[Order],
        base_profiles: Sequence[Sequence[Order]],
        depth: int,
    ) -> float:
        if not base_profiles:
            return self._resolve_and_score(
                state, power, unit_order, combo, tuple(), depth
            )

        total = 0.0
        for profile in base_profiles:
            total += self._resolve_and_score(
                state, power, unit_order, combo, profile, depth
            )
        return total / float(len(base_profiles))

    def _resolve_and_score(
        self,
        state: GameState,
        power: Power,
        unit_order: Sequence[str],
        combo: Sequence[Order],
        opponent_orders: Sequence[Order],
        depth: int,
    ) -> float:
        assigned: set[str] = set(unit_order)
        orders: List[Order] = list(combo)

        for order in opponent_orders:
            if order.unit.loc in assigned:
                continue
            orders.append(order)
            assigned.add(order.unit.loc)

        # The adjudicator automatically supplies hold orders for any remaining
        # units, so we can avoid generating them manually. This keeps the
        # resolution step lean, which is important because it runs for every
        # candidate joint order we evaluate.
        next_state, _ = Adjudicator(state).resolve(orders)
        observation: Optional[Any] = None
        current_score = self._evaluate_state(next_state, observation, power)

        if depth <= 1:
            return current_score

        candidate_map = self._build_candidate_order_map(next_state, power)
        if not candidate_map:
            return current_score

        _, future_score = self._select_best_orders(
            next_state, power, candidate_map, depth=depth - 1
        )
        return current_score + self.rollout_discount * future_score

    def _evaluate_state(
        self,
        state: GameState,
        observation: Optional[Any],
        power: Power,
    ) -> float:
        threatened = float(state.centers_threatened(power))

        sc_control = float(
            sum(1 for controller in state.supply_center_control.values() if controller == power)
        )

        # Always use state-based unit count since observation is not available
        unit_presence = float(sum(1 for unit in state.units.values() if unit.power == power))

        return (
            self.unit_weight * unit_presence
            + self.supply_center_weight * sc_control
            - self.threatened_penalty * threatened
        )


class ObservationBestResponseAgent(Agent):
    """Agent that selects orders via an observation-based best response policy."""

    def __init__(
        self,
        power: Power,
        *,
        policy: SampledBestResponsePolicy | None = None,
    ) -> None:
        super().__init__(power)
        self._policy = policy or SampledBestResponsePolicy()

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        return self._policy.plan_orders(state, self.power)

    def plan_builds(self, state: GameState, build_count: int) -> List[Tuple[str, UnitType]]:
        return self._policy.plan_builds(state, self.power, build_count)

__all__ = ["ObservationBestResponseAgent", "SampledBestResponsePolicy"]
