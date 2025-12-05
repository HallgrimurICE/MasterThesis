# sl_agent.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..state import GameState
from ..types import Order, OrderType, Power
from .base import Agent

from policytraining.run_sl import make_sl_policy
from ..value_estimation import save, sl_state_value

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
        - read the 'values' entry from the returned info dict via sl_state_value
          helper.
        """
        return sl_state_value(state, power, policy=self._policy)

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

    def _build_rollout_agents(self, state: GameState) -> Dict[Power, Agent]:
        return {power: _SlPolicyAgent(power, self._policy) for power in state.powers}


class _SlPolicyAgent(Agent):
    """Lightweight wrapper to sample orders directly from the SL policy."""

    def __init__(self, power: Power, policy):
        super().__init__(power)
        self._policy = policy

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        observation = build_observation(state, last_actions=[])
        legal_actions = list(legal_actions_from_state(state))
        if not legal_actions:
            return []

        powers = sorted(state.powers, key=str)
        try:
            my_index = powers.index(self.power)
        except ValueError:
            return []

        slots_list: Sequence[int] = list(range(len(powers)))
        actions, _ = self._policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_actions,
        )
        action_indices = list(actions[my_index])
        return decode_actions_to_orders(state=state, power=self.power, action_indices=action_indices)


class DeepMindSaveAgent(DeepMindSlAgent):
    """DeepMind SL agent that scores candidates with SAVE rollouts."""

    def _best_response_action_indices(self, state: GameState) -> List[int]:
        powers: List[Power] = sorted(state.powers, key=str)
        legal_actions = list(legal_actions_from_state(state))
        if not legal_actions:
            return []

        my_power = self.power
        my_index = powers.index(my_power)
        slots_list: List[int] = list(range(len(powers)))

        best_value = float("-inf")
        best_candidate: Optional[List[int]] = None

        K = getattr(self, "_k_candidates", 8)
        N = getattr(self, "_action_rollouts", 4)

        observation = build_observation(state, last_actions=[])
        rollout_agents = self._build_rollout_agents(state)

        for _ in range(K):
            joint_actions_array, _ = self._policy.actions(
                slots_list=slots_list,
                observation=observation,
                legal_actions=legal_actions,
            )
            my_candidate_indices = list(joint_actions_array[my_index])
            candidate_orders = decode_actions_to_orders(
                state=state,
                power=my_power,
                action_indices=my_candidate_indices,
            )

            score = save(
                initial_state=state,
                focal_power=my_power,
                candidate_orders=candidate_orders,
                agents=rollout_agents,
                n_rollouts=N,
                horizon=1,
                value_fn=sl_state_value,
                value_kwargs={"policy": self._policy},
            )

            if score > best_value:
                best_value = score
                best_candidate = my_candidate_indices

        return best_candidate or []

class _ContractAwareSlPolicyAgent(_SlPolicyAgent):
    def __init__(self, power: Power, policy, contracts):
        super().__init__(power, policy)
        self._contracts = contracts

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        powers = sorted(state.powers, key=str)
        my_index = powers.index(self.power)
        all_legal = list(legal_actions_from_state(state))
        legal_actions = restrict_actions_for_power(
            self.power,
            all_legal[my_index],
            self._contracts,
        )
        if len(legal_actions) == 0:
            return []
        obs = build_observation(state, last_actions=[])
        slots = list(range(len(powers)))
        legal_by_power = []
        for p in powers:
            allowed = restrict_actions_for_power(
                p,
                all_legal[powers.index(p)],
                self._contracts,
            )
            legal_by_power.append(np.asarray(allowed, dtype=np.int64))
        actions, _ = self._policy.actions(slots_list=slots, observation=obs, legal_actions=legal_by_power)
        return decode_actions_to_orders(
            state=state,
            power=self.power,
            action_indices=list(actions[my_index]),
        )

class _ContractAwareSlPolicyAgent(_SlPolicyAgent):
    def __init__(self, power: Power, policy, contracts):
        super().__init__(power, policy)
        self._contracts = contracts

    def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
        del round_index
        powers = sorted(state.powers, key=str)
        my_index = powers.index(self.power)
        all_legal = list(legal_actions_from_state(state))
        legal_actions = restrict_actions_for_power(
            self.power,
            list(all_legal[my_index]),
            self._contracts,
        )
        if len(legal_actions) == 0:
            return []
        obs = build_observation(state, last_actions=[])
        slots = list(range(len(powers)))
        legal_by_power = []
        for p in powers:
            allowed = restrict_actions_for_power(
                p,
                list(all_legal[powers.index(p)]),
                self._contracts,
            )
            legal_by_power.append(np.asarray(list(allowed), dtype=np.int64))
        actions, _ = self._policy.actions(slots_list=slots, observation=obs, legal_actions=legal_by_power)
        return decode_actions_to_orders(
            state=state,
            power=self.power,
            action_indices=list(actions[my_index]),
        )


class DeepMindNegotiatorAgent(DeepMindSaveAgent):
    def __init__(self, *args, rss_rollouts: int = 4, relationship_decay: float = 0.95, relationship_step: float = 0.2, propose_threshold: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._rss_rollouts = rss_rollouts
        self._relationship: Dict[Power, float] = {}
        self._last_seen_state: Optional[GameState] = None
        self._relationship_decay = relationship_decay
        self._relationship_step = relationship_step
        self._propose_threshold = propose_threshold

    def issue_orders(self, state: GameState) -> List[Order]:
        self._update_relationships(state)
        return super().issue_orders(state)

    def _policy_fn(self, powers: Sequence[Power], idx: int):
        def fn(state: GameState, _power: Power, legal_actions, restricted):
            obs = build_observation(state, last_actions=[])
            slots = list(range(len(powers)))
            joint_legal = []
            for p in powers:
                allowed = restricted.get(p) if restricted and p in restricted else legal_actions[p]
                joint_legal.append(np.asarray(allowed, dtype=np.int64))
            actions, _ = self._policy.actions(slots_list=slots, observation=obs, legal_actions=joint_legal)
            return list(actions[idx])
        return fn

    def _build_rollout_agents(self, state: GameState, contracts) -> Dict[Power, Agent]:
        return {p: _ContractAwareSlPolicyAgent(p, self._policy, contracts) for p in state.powers}

    def _best_response_action_indices(self, state: GameState) -> List[int]:
        powers = sorted(state.powers, key=str)
        self._ensure_relationships(powers)
        raw_legal = list(legal_actions_from_state(state))
        if not raw_legal:
            return []

        # RSS proposals -> contracts
        legal_map = {p: list(raw_legal[i]) for i, p in enumerate(powers)}
        policy_fns = {p: self._policy_fn(powers, i) for i, p in enumerate(powers)}
        proposals = {
            p: run_rss_for_power(
                state=state,
                power=p,
                powers=powers,
                legal_actions=legal_map,
                policy_fns=policy_fns,
                value_fn=lambda s, pow=p: self._state_value(s, pow),
                step_fn=self._step_state,
                rollouts=self._rss_rollouts,
            )
            for p in powers
        }
        # Apply simple trust gating: refuse to propose peace to partners below threshold.
        if self.power in proposals:
            proposals[self.power] = {
                other for other in proposals[self.power] if self._relationship.get(other, 0.0) >= self._propose_threshold
            }

        contracts = compute_active_contracts(state, powers, legal_map, proposals)

        # Restrict legal actions under contracts
        restricted = [
            np.asarray(restrict_actions_for_power(p, raw_legal[i], contracts), dtype=np.int64)
            for i, p in enumerate(powers)
        ]

        # SAVE best-response with restricted actions and contract-aware rollouts
        observation = build_observation(state, last_actions=[])
        rollout_agents = self._build_rollout_agents(state, contracts)
        my_index = powers.index(self.power)
        K = getattr(self, "_k_candidates", 8)
        N = getattr(self, "_action_rollouts", 4)

        best_val, best_candidate = float("-inf"), None
        slots = list(range(len(powers)))
        for _ in range(K):
            joint_actions, _ = self._policy.actions(
                slots_list=slots,
                observation=observation,
                legal_actions=restricted,
            )
            my_candidate = list(joint_actions[my_index])
            candidate_orders = decode_actions_to_orders(state, self.power, my_candidate)
            score = save(
                initial_state=state,
                focal_power=self.power,
                candidate_orders=candidate_orders,
                agents=rollout_agents,
                n_rollouts=N,
                horizon=1,
                value_fn=sl_state_value,
                value_kwargs={"policy": self._policy},
            )
            score -= self._trust_penalty(state, candidate_orders)
            if score > best_val:
                best_val, best_candidate = score, my_candidate

        return best_candidate or []

    def _ensure_relationships(self, powers: Sequence[Power]) -> None:
        for p in powers:
            if p == self.power:
                continue
            self._relationship.setdefault(p, 0.0)

    def _update_relationships(self, state: GameState) -> None:
        powers = sorted(state.powers, key=str)
        self._ensure_relationships(powers)
        if self._last_seen_state is None:
            self._last_seen_state = state.copy()
            return

        # Mild decay toward neutral.
        for p, val in list(self._relationship.items()):
            self._relationship[p] = max(-1.0, min(1.0, val * self._relationship_decay))

        prev_sc = self._last_seen_state.supply_center_control
        curr_sc = state.supply_center_control
        for province, controller in prev_sc.items():
            if controller != self.power:
                continue
            current_controller = curr_sc.get(province)
            if current_controller is None or current_controller == self.power:
                continue
            # Losing a center to another power lowers trust.
            self._adjust_relationship(current_controller, -self._relationship_step)

        # If an enemy now occupies a province we previously occupied, treat it as hostile.
        for province, unit in self._last_seen_state.units.items():
            if unit.power != self.power:
                continue
            occupying = state.units.get(province)
            if occupying is not None and occupying.power != self.power:
                self._adjust_relationship(occupying.power, -self._relationship_step * 0.5)

        self._last_seen_state = state.copy()

    def _adjust_relationship(self, power: Power, delta: float) -> None:
        if power == self.power:
            return
        self._relationship[power] = max(-1.0, min(1.0, self._relationship.get(power, 0.0) + delta))

    def _trust_penalty(self, state: GameState, candidate_orders: Sequence[Order]) -> float:
        """Discourage attacking high-trust partners."""
        penalty = 0.0
        for order in candidate_orders:
            if order.type == OrderType.MOVE and order.target:
                target_unit = state.units.get(order.target)
                target_owner = target_unit.power if target_unit else state.supply_center_control.get(order.target)
                penalty += self._attack_penalty(target_owner)
            elif order.type == OrderType.SUPPORT and order.support_target:
                target_unit = state.units.get(order.support_target)
                target_owner = target_unit.power if target_unit else state.supply_center_control.get(order.support_target)
                penalty += 0.5 * self._attack_penalty(target_owner)
        return penalty

    def _attack_penalty(self, owner: Optional[Power]) -> float:
        if owner is None or owner == self.power:
            return 0.0
        rel = self._relationship.get(owner, 0.0)
        return rel * 0.5 if rel > 0.0 else 0.0
