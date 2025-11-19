# sl_agent.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..state import GameState
from ..types import Order, Power
from .base import Agent

from policytraining.run_sl import make_sl_policy

# These are the DeepMind-style helpers you are (or will be) writing.
# They should live under diplomacy/deepmind/ and map your GameState to DM obs/actions.
from ..deepmind.build_observation import build_observation
from ..deepmind.actions import (
    legal_actions_from_state,     # (state: GameState) -> Sequence[np.ndarray]
    decode_actions_to_orders,     # (state, power, action_indices) -> List[Order]
)
from ..adjudication import Adjudicator
from ..negotiation.contracts import Contract, restrict_actions_for_power
from ..negotiation.rss import compute_active_contracts, run_rss_for_power



class DeepMindSlAgent(Agent):
    """Agent wrapper that controls a single power via DeepMind's SL policy."""

    def __init__(
        self,
        power: Power,
        sl_params_path: str,
        rng_seed: int = 0,
        temperature: float = 0.1,
    ):
        super().__init__(power)
        self._policy = make_sl_policy(sl_params_path, rng_seed=rng_seed)
        self._temperature = temperature  # currently configured inside make_sl_policy

    # --------- main per-round hook used by the game loop ----------

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        """Plan movement orders for this agent's power in the current phase."""

        del round_index  # the SL policy is stateless, so we ignore the round counter

        # 1) Build DeepMind observation from your engine state.
        #
        # You already have tests for this in `test_deepmind_observation.py`,
        # which call `diplomacy.deepmind.build_observation(state, last_actions=[...])`.
        #
        # Here we assume last_actions=[] for simplicity, but you can thread in
        # a history of last action indices if you want to be fancy later.
        observation = build_observation(state, last_actions=[])

        # 2) Build DM-style legal actions: list of arrays, one per player.
        # The test file suggests you are designing a mapping layer under
        # `diplomacy.deepmind`. Here we just call it.
        legal_actions = legal_actions_from_state(state)

        # Make sure we know which slot index corresponds to `self.power`.
        # You should have a consistent ordering of powers (e.g., sorted by name).
        powers: List[Power] = sorted(state.powers, key=str)
        num_players = len(powers)
        assert len(legal_actions) == num_players, "legal_actions must align with powers"

        try:
            my_index = powers.index(self.power)
        except ValueError:
            raise ValueError(f"Agent power {self.power} not in state.powers={powers}")

        # 3) Ask the SL policy for actions for *all* players (or just this one).
        slots_list: Sequence[int] = list(range(num_players))  # all players
        # If you only want the policy's suggestion for this power, you can also use:
        # slots_list = [my_index]

        actions, info = self._policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_actions,
        )

        # `actions` is a list (per slot) of sequences of action indices.
        my_action_indices = list(actions[my_index])

        # 4) Convert action indices for this power back to engine Orders.
        orders: List[Order] = decode_actions_to_orders(
            state=state,
            power=self.power,
            action_indices=my_action_indices,
        )

        return orders

    # --------- optional: handle retreats & builds if you want ----------

    def _plan_retreat_orders(self, state: GameState, retreat_index: int) -> List[Order]:
        # For now, just hold / disband or delegate to some heuristic.
        # You can make this smarter later if you like.
        return []

    def plan_builds(self, state: GameState, build_count: int):
        # For now, do nothing (no builds). Or implement a simple heuristic.
        return []


class BaselineNegotiatorAgent(DeepMindSlAgent):
    """Baseline SL agent augmented with RSS-style Peace negotiations.

    The agent follows the Round-Simultaneous Signaling (RSS) protocol from the
    DeepMind negotiation work: it simulates counterfactual Peace contracts for
    every pair of powers, exchanges proposals, and enforces mutually agreed
    contracts by masking the legal action space before sampling orders.
    """

    _negotiation_cache: Dict[Tuple[Any, ...], Sequence["Contract"]] = {}

    def __init__(
        self,
        power: Power,
        sl_params_path: str,
        *,
        rng_seed: int = 0,
        temperature: float = 0.1,
        mc_rollouts: int = 4,
    ):
        super().__init__(power, sl_params_path, rng_seed=rng_seed, temperature=temperature)
        self._mc_rollouts = mc_rollouts

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        print(
            f"[BaselineNegotiator][{self.power}] Round {round_index}: phase {state.phase.name}"
        )

        powers = sorted(state.powers, key=str)
        legal_actions_map = self._legal_actions_map(state, powers)
        policy_fns = self._build_policy_functions(powers)

        value_fn = self._state_value
        step_fn = self._step_state

        active_contracts = self._active_contracts_for_state(
            state,
            powers,
            legal_actions_map,
            policy_fns,
            value_fn,
            step_fn,
        )
        if active_contracts:
            for contract in active_contracts:
                if self.power not in (contract.player_i, contract.player_j):
                    continue
                print(
                    "[BaselineNegotiator] Round"
                    f" {round_index} {state.phase.name}: Peace between"
                    f" {contract.player_i} and {contract.player_j}"
                )
        my_contracts = [c for c in active_contracts if self.power in (c.player_i, c.player_j)]
        restrictions_map: Optional[Dict[Power, Sequence[int]]] = None
        if my_contracts:
            restricted = restrict_actions_for_power(
                self.power, legal_actions_map[self.power], my_contracts
            )
            restrictions_map = {self.power: tuple(int(a) for a in restricted)}

        my_action_indices = self._sample_action_indices(
            state,
            focus_power=self.power,
            powers=powers,
            legal_actions=legal_actions_map,
            restricted_actions=restrictions_map,
        )
        return decode_actions_to_orders(state=state, power=self.power, action_indices=my_action_indices)

    # ------------------------------------------------------------------
    # Helper methods used by RSS + Peace logic

    @staticmethod
    def _state_signature(state: GameState) -> Tuple[Any, ...]:
        unit_signature = tuple(
            sorted((province, unit.power, unit.unit_type.name) for province, unit in state.units.items())
        )
        retreat_signature = tuple(
            sorted((province, unit.power, unit.unit_type.name) for province, unit in state.pending_retreats.items())
        )
        controller_signature = tuple(sorted(state.supply_center_control.items()))
        return (state.phase.name, unit_signature, retreat_signature, controller_signature)

    def _active_contracts_for_state(
        self,
        state: GameState,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        policy_fns: Mapping[Power, Callable[
            [GameState, Power, Mapping[Power, Sequence[int]], Optional[Mapping[Power, Sequence[int]]]],
            Sequence[int],
        ]],
        value_fn: Callable[[GameState, Power], float],
        step_fn: Callable[[GameState, Mapping[Power, Sequence[int]]], GameState],
    ) -> Sequence["Contract"]:
        signature = self._state_signature(state)
        cached = self._negotiation_cache.get(signature)
        if cached is not None:
            return cached

        proposals: Dict[Power, Set[Power]] = {}
        for power in powers:
            proposals[power] = run_rss_for_power(
                state=state,
                power=power,
                powers=powers,
                legal_actions=legal_actions,
                policy_fns=policy_fns,
                value_fn=value_fn,
                step_fn=step_fn,
                rollouts=self._mc_rollouts,
            )

        active_contracts = compute_active_contracts(state, powers, legal_actions, proposals)
        self._negotiation_cache[signature] = active_contracts
        return active_contracts

    def _legal_actions_map(
        self, state: GameState, powers: Sequence[Power]
    ) -> Dict[Power, Sequence[int]]:
        legal_actions = legal_actions_from_state(state)
        if len(legal_actions) != len(powers):
            raise ValueError("legal actions do not match power list")
        return {
            power: tuple(int(action) for action in legal_actions[idx])
            for idx, power in enumerate(powers)
        }

    def _build_policy_functions(
        self, powers: Sequence[Power]
    ) -> Dict[Power, Callable[[GameState, Power, Mapping[Power, Sequence[int]], Optional[Mapping[Power, Sequence[int]]]], Sequence[int]]]:
        functions: Dict[Power, Callable[[GameState, Power, Mapping[Power, Sequence[int]], Optional[Mapping[Power, Sequence[int]]]], Sequence[int]]] = {}

        for power in powers:
            functions[power] = self._make_policy_fn(power, powers)
        return functions

    def _make_policy_fn(
        self, acting_power: Power, powers: Sequence[Power]
    ) -> Callable[[GameState, Power, Mapping[Power, Sequence[int]], Optional[Mapping[Power, Sequence[int]]]], Sequence[int]]:

        def policy_fn(
            state: GameState,
            power: Power,
            legal_actions: Mapping[Power, Sequence[int]],
            restricted: Optional[Mapping[Power, Sequence[int]]],
        ) -> Sequence[int]:
            del power  # the closure already knows which power it represents
            return self._sample_action_indices(
                state,
                focus_power=acting_power,
                powers=powers,
                legal_actions=legal_actions,
                restricted_actions=restricted,
            )

        return policy_fn

    def _sample_action_indices(
        self,
        state: GameState,
        focus_power: Power,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    ) -> List[int]:
        observation = build_observation(state, last_actions=[])
        legal_arrays: List[np.ndarray] = []
        for power in powers:
            actions = legal_actions[power]
            allowed = restricted_actions.get(power) if restricted_actions else None
            filtered = self._apply_restrictions(actions, allowed)
            legal_arrays.append(np.asarray(filtered, dtype=np.int64))

        slots_list: Sequence[int] = list(range(len(powers)))
        actions, _ = self._policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_arrays,
        )
        return list(actions[powers.index(focus_power)])

    def _apply_restrictions(
        self,
        legal_actions: Sequence[int],
        allowed_subset: Optional[Sequence[int]],
    ) -> Sequence[int]:
        legal = [int(action) for action in legal_actions]
        if not allowed_subset:
            return legal
        allowed = {int(action) for action in allowed_subset}
        filtered = [action for action in legal if action in allowed]
        return filtered or legal

    def _state_value(self, state: GameState, power: Power) -> float:
        observation = build_observation(state, last_actions=[])
        legal_actions = legal_actions_from_state(state)
        if not legal_actions:
            return 0.0
        powers = sorted(state.powers, key=str)
        slots_list: Sequence[int] = list(range(len(powers)))
        _, info = self._policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_actions,
        )
        values = info.get("values") if isinstance(info, dict) else None
        if values is None:
            return 0.0
        try:
            return float(values[powers.index(power)])
        except ValueError:
            return 0.0

    def _step_state(
        self, state: GameState, joint_actions: Mapping[Power, Sequence[int]]
    ) -> GameState:
        orders: List[Order] = []
        for power, action_indices in joint_actions.items():
            orders.extend(decode_actions_to_orders(state, power, action_indices))
        next_state, _ = Adjudicator(state).resolve(orders)
        return next_state
