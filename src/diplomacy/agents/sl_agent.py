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
        k_candidates: int = 8,
        action_rollouts: int = 4,
    ):
        super().__init__(power)
        self._policy = make_sl_policy(sl_params_path, rng_seed=rng_seed)
        # Temperature is currently configured inside make_sl_policy, but we keep
        # the field for future use / inspection.
        self._temperature = temperature

        # Parameters for sampled best-response search.
        self._k_candidates = k_candidates       # K: number of candidate moves
        self._action_rollouts = action_rollouts # N: rollouts per candidate

    # ------------------------------------------------------------------------
    # Public planning API used by simulation.run_rounds_with_agents
    # ------------------------------------------------------------------------
    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        """Plan a set of orders for this power in the given state.

          - sample K candidate actions for this power
          - for each candidate, run N rollouts with other powers sampled
            from the policy
          - pick the candidate with highest average value.
        """
        del round_index  # not used in this agent

        # Use best-response search to pick our action indices.
        my_action_indices = self._best_response_action_indices(state)

        orders: List[Order] = decode_actions_to_orders(
            state=state,
            power=self.power,
            action_indices=my_action_indices,
        )
        return orders

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------
    def _step_state(
        self,
        state: GameState,
        joint_actions: Mapping[Power, Sequence[int]],
    ) -> GameState:
        """Apply encoded joint actions and return the next GameState.

        joint_actions: mapping from Power -> sequence of encoded actions
                       (the 64-bit DeepMind action integers).
        """
        orders: List[Order] = []
        for power, action_indices in joint_actions.items():
            orders.extend(
                decode_actions_to_orders(
                    state=state,
                    power=power,
                    action_indices=action_indices,
                )
            )

        # Adjudicator.resolve returns (next_state, titles_history)
        next_state, _ = Adjudicator(state).resolve(orders)
        return next_state

    def _state_value(self, state: GameState, power: Power) -> float:
        """Return this power's scalar value estimate V(s) from the SL policy.

        We follow the same pattern as in BaselineNegotiatorAgent:
        - call policy.actions(...) to run the network
        - read the 'values' entry from the returned info dict
        - index into it using the power's slot.
        """
        observation = build_observation(state, last_actions=[])
        legal_actions = legal_actions_from_state(state)
        # If there are no legal actions (should be rare), just return 0.
        if not legal_actions:
            return 0.0

        powers: List[Power] = sorted(state.powers, key=str)
        slots_list: Sequence[int] = list(range(len(powers)))

        # We don't actually care about the sampled actions here, only 'info'.
        _, info = self._policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_actions,
        )

        values = info.get("values") if isinstance(info, dict) else None
        if values is None:
            return 0.0

        try:
            # values is assumed to be a 1D array aligned with 'powers'.
            return float(values[powers.index(power)])
        except (ValueError, IndexError, TypeError):
            return 0.0

    def _best_response_action_indices(self, state: GameState) -> List[int]:
        """Sample K candidate actions for me and evaluate each with N rollouts.

        Algorithm:
          - Let powers be the sorted list of powers, my_index our index.
          - legal_actions[p] is the list of encoded actions for power p.
          - For k in 1..K:
              * sample a joint action profile from the policy
              * take my part as candidate 'a_k'
              * estimate its value with N rollouts:
                  - fix my action to 'a_k'
                  - resample others from policy
                  - step one turn, evaluate V(next_state) for me
              * keep the candidate with highest average V.
        """
        powers: List[Power] = sorted(state.powers, key=str)
        num_players = len(powers)
        my_power = self.power
        my_index = powers.index(my_power)

        # 1) Get full legal actions for all powers in this state.
        legal_actions = list(legal_actions_from_state(state))

        # Precompute observation once for this state.
        observation = build_observation(state, last_actions=[])

        best_value = float("-inf")
        best_candidate: Optional[List[int]] = None

        K = getattr(self, "_k_candidates", 8)
        N = getattr(self, "_action_rollouts", 4)

        slots_list: List[int] = list(range(num_players))  # one slot per power

        for _ in range(K):
            # 2) Sample one joint-action profile; take my action as the candidate.
            joint_actions_array, _ = self._policy.actions(
                slots_list=slots_list,
                observation=observation,
                legal_actions=legal_actions,
            )
            my_candidate_indices = list(joint_actions_array[my_index])

            # 3) Evaluate this candidate with N Monte-Carlo rollouts.
            total_v = 0.0

            for _ in range(N):
                # Build legal_actions for rollout:
                rollout_legal_actions = list(legal_actions)

                # For my power, restrict to exactly this candidate.
                rollout_legal_actions[my_index] = np.asarray(
                    my_candidate_indices, dtype=np.int64
                )

                # Resample other powers subject to the restriction on me.
                rollout_actions_array, _ = self._policy.actions(
                    slots_list=slots_list,
                    observation=observation,
                    legal_actions=rollout_legal_actions,
                )

                # Convert to mapping Power -> List[int]
                joint_actions: Dict[Power, List[int]] = {}
                for slot, p in enumerate(powers):
                    joint_actions[p] = list(rollout_actions_array[slot])

                # Step the state and evaluate my value at the next state.
                next_state = self._step_state(state, joint_actions)
                v = self._state_value(next_state, my_power)
                total_v += v

            avg_v = total_v / float(N)

            if avg_v > best_value:
                best_value = avg_v
                best_candidate = my_candidate_indices

        assert best_candidate is not None
        return best_candidate



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
        self._joint_sample_cache: Dict[
            Tuple[int, int, Optional[int]], Tuple[Dict[Power, Sequence[int]], int]
        ] = {}

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
        self._joint_sample_cache.clear()
        signature = self._state_signature(state)
        cached = self._negotiation_cache.get(signature)
        if cached is not None:
            return cached

        value_cache: Dict[
            Tuple[Tuple[Any, ...], Tuple[Tuple[Power, Tuple[int, ...]], ...]], Mapping[Power, float]
        ] = {}
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
                value_cache=value_cache,
                state_signature=signature,
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
            return self._cached_action_indices(
                state,
                focus_power=acting_power,
                powers=powers,
                legal_actions=legal_actions,
                restricted_actions=restricted,
            )

        return policy_fn

    def _cached_action_indices(
        self,
        state: GameState,
        focus_power: Power,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    ) -> List[int]:
        key = self._joint_sample_cache_key(state, legal_actions, restricted_actions)
        cached = self._joint_sample_cache.get(key)
        if cached is None:
            joint = self._sample_joint_actions(state, powers, legal_actions, restricted_actions)
            cached = (joint, len(powers))
        joint_actions, remaining = cached
        actions = list(joint_actions[focus_power])
        remaining -= 1
        if remaining <= 0:
            self._joint_sample_cache.pop(key, None)
        else:
            self._joint_sample_cache[key] = (joint_actions, remaining)
        return actions

    def _sample_action_indices(
        self,
        state: GameState,
        focus_power: Power,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    ) -> List[int]:
        joint = self._sample_joint_actions(
            state,
            powers=powers,
            legal_actions=legal_actions,
            restricted_actions=restricted_actions,
        )
        return list(joint[focus_power])

    def _sample_joint_actions(
        self,
        state: GameState,
        powers: Sequence[Power],
        legal_actions: Mapping[Power, Sequence[int]],
        restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    ) -> Dict[Power, Sequence[int]]:
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
        return {power: list(actions[idx]) for idx, power in enumerate(powers)}

    def _joint_sample_cache_key(
        self,
        state: GameState,
        legal_actions: Mapping[Power, Sequence[int]],
        restricted_actions: Optional[Mapping[Power, Sequence[int]]],
    ) -> Tuple[int, int, Optional[int]]:
        restricted_id = id(restricted_actions) if restricted_actions is not None else None
        return (id(state), id(legal_actions), restricted_id)

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
