from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..adjudication import Adjudicator
from ..deepmind.actions import (
    decode_action_to_order,
    decode_actions_to_orders,
    legal_actions_from_state,
)
from ..negotiation.contracts import Contract, restrict_actions_for_power
from ..negotiation.rss import compute_active_contracts, run_rss_for_power
from ..orders import hold
from ..state import GameState
from ..types import Order, Power, UnitType
from .base import Agent
from .best_response import SampledBestResponsePolicy


class HeuristicNegotiatorAgent(Agent):
    """Greedy heuristic agent that negotiates with RSS + ToM depth."""

    def __init__(
        self,
        power: Power,
        *,
        policy: Optional[SampledBestResponsePolicy] = None,
        rss_rollouts: int = 1,
        tom_depth: int = 1,
        rng: Optional[random.Random] = None,
        unit_weight: float = 1.0,
        supply_center_weight: float = 5.0,
        threatened_penalty: float = 2.0,
        log_negotiation: bool = False,
    ) -> None:
        super().__init__(power)
        self._policy = policy or SampledBestResponsePolicy()
        self._rss_rollouts = rss_rollouts
        self._tom_depth = tom_depth
        self._rng = rng or random.Random()
        self._unit_weight = unit_weight
        self._supply_center_weight = supply_center_weight
        self._threatened_penalty = threatened_penalty
        self._log_negotiation = log_negotiation

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        contracts = self._compute_contracts(state)
        planned_orders = self._policy.plan_orders(state, self.power)
        if not contracts:
            return planned_orders
        return self._apply_contracts(state, planned_orders, contracts)

    def plan_builds(self, state: GameState, build_count: int) -> List[tuple[str, UnitType]]:
        return self._policy.plan_builds(state, self.power, build_count)

    def _compute_contracts(self, state: GameState) -> Sequence[Contract]:
        powers = sorted(state.powers, key=str)
        legal_arrays = list(legal_actions_from_state(state))
        legal_map = {p: list(legal_arrays[i]) for i, p in enumerate(powers)}
        policy_fns = {
            power: self._policy_fn(power)
            for power in powers
        }

        proposals: Dict[Power, Set[Power]] = {}
        for power in powers:
            proposals[power] = run_rss_for_power(
                state=state,
                power=power,
                powers=powers,
                legal_actions=legal_map,
                policy_fns=policy_fns,
                value_fn=self._heuristic_value,
                step_fn=self._step_state_from_actions,
                rollouts=self._rss_rollouts,
                tom_depth=self._tom_depth,
            )
        contracts = compute_active_contracts(state, powers, legal_map, proposals)
        if self._log_negotiation:
            self._log_contracts(state, proposals, contracts)
        return contracts

    def _policy_fn(self, power: Power):
        def fn(
            state: GameState,
            _power: Power,
            legal_actions: Mapping[Power, Iterable[int]],
            restricted_actions: Optional[Mapping[Power, Iterable[int]]] = None,
        ) -> List[int]:
            allowed = list(legal_actions[power])
            if restricted_actions and power in restricted_actions:
                allowed = list(restricted_actions[power])
            return self._sample_actions_for_power(state, power, allowed)

        return fn

    def _sample_actions_for_power(
        self,
        state: GameState,
        power: Power,
        allowed_actions: Sequence[int],
    ) -> List[int]:
        unit_locs = {unit.loc for unit in state.units.values() if unit.power == power}
        options_by_loc: Dict[str, List[int]] = {loc: [] for loc in unit_locs}
        for encoded in allowed_actions:
            order = decode_action_to_order(state, power, int(encoded))
            if order is None:
                continue
            loc = order.unit.loc
            if loc not in options_by_loc:
                continue
            options_by_loc[loc].append(int(encoded))

        chosen: List[int] = []
        for loc, options in options_by_loc.items():
            if not options:
                continue
            chosen.append(self._rng.choice(options))
        return chosen

    def _heuristic_value(self, state: GameState, power: Power) -> float:
        threatened = float(state.centers_threatened(power))
        sc_control = float(
            sum(1 for controller in state.supply_center_control.values() if controller == power)
        )
        unit_presence = float(sum(1 for unit in state.units.values() if unit.power == power))
        return (
            self._unit_weight * unit_presence
            + self._supply_center_weight * sc_control
            - self._threatened_penalty * threatened
        )

    def _step_state_from_actions(
        self,
        state: GameState,
        joint_actions: Mapping[Power, Iterable[int]],
    ) -> GameState:
        orders: List[Order] = []
        for power, action_indices in joint_actions.items():
            orders.extend(
                decode_actions_to_orders(
                    state=state,
                    power=power,
                    action_indices=list(action_indices),
                )
            )
        next_state, _ = Adjudicator(state).resolve(orders)
        return next_state

    def _apply_contracts(
        self,
        state: GameState,
        planned_orders: Sequence[Order],
        contracts: Sequence[Contract],
    ) -> List[Order]:
        powers = sorted(state.powers, key=str)
        my_index = powers.index(self.power)
        all_legal = list(legal_actions_from_state(state))
        allowed_actions = restrict_actions_for_power(
            self.power,
            list(all_legal[my_index]),
            contracts,
        )
        allowed_by_loc: Dict[str, Set[Order]] = {}
        for encoded in allowed_actions:
            order = decode_action_to_order(state, self.power, int(encoded))
            if order is None:
                continue
            allowed_by_loc.setdefault(order.unit.loc, set()).add(order)

        adjusted_orders: List[Order] = []
        planned_by_loc = {order.unit.loc: order for order in planned_orders}
        for unit in state.units.values():
            if unit.power != self.power:
                continue
            planned = planned_by_loc.get(unit.loc)
            allowed_orders = allowed_by_loc.get(unit.loc)
            if planned and (not allowed_orders or planned in allowed_orders):
                adjusted_orders.append(planned)
            else:
                adjusted_orders.append(hold(unit))
        return adjusted_orders

    def _log_contracts(
        self,
        state: GameState,
        proposals: Mapping[Power, Set[Power]],
        contracts: Sequence[Contract],
    ) -> None:
        phase = state.phase.name
        powers = sorted(state.powers, key=str)
        print(f"[heuristic_negotiator] Phase={phase} Negotiation proposals:")
        for power in powers:
            partners = ", ".join(str(p) for p in sorted(proposals.get(power, set()), key=str))
            print(f"  {power}: {partners or '(none)'}")
        if contracts:
            print("[heuristic_negotiator] Active contracts:")
            for contract in contracts:
                print(f"  {contract.power_i} <-> {contract.power_j}")
        else:
            print("[heuristic_negotiator] Active contracts: (none)")


__all__ = ["HeuristicNegotiatorAgent"]
