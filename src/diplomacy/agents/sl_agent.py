# sl_agent.py

from __future__ import annotations

from typing import List, Dict, Any, Sequence

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
