from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .base import Agent
from .best_response import SampledBestResponsePolicy
from .random import RandomAgent
from ..types import Order, Power
from ..value_estimation import save, sl_state_value, ValueFn


OpponentFactory = Callable[[Power], Agent]


class _PolicyAgent(Agent):
    """Thin wrapper to use a policy's plan_orders without invoking SAVE."""

    def __init__(self, power: Power, policy: SampledBestResponsePolicy):
        super().__init__(power)
        self._policy = policy

    def _plan_orders(self, state, round_index: int) -> List[Order]:
        return self._policy.plan_orders(state, self.power)


class SaveBestResponseAgent(Agent):
    """Best-response agent that scores candidates with SAVE + a configurable value function."""

    def __init__(
        self,
        power: Power,
        *,
        opponent_factory: Optional[OpponentFactory] = None,
        value_fn: ValueFn = sl_state_value,
        value_kwargs: Optional[dict] = None,
        n_rollouts: int = 2,
        horizon: int = 1,
        max_candidates: Optional[int] = None,
        policy: Optional[SampledBestResponsePolicy] = None,
        rng_seed: int = 0,
    ) -> None:
        super().__init__(power)
        self._opponent_factory = opponent_factory or (
            lambda p: RandomAgent(p, hold_probability=0.2, rng=random.Random(rng_seed + hash(p)))
        )
        self._value_fn = value_fn
        self._value_kwargs = value_kwargs or {}
        self._n_rollouts = n_rollouts
        self._horizon = horizon
        self._policy = policy or SampledBestResponsePolicy(rollout_limit=max_candidates or 8, rng=random.Random(rng_seed))
        self._max_candidates = max_candidates or self._policy.rollout_limit
        self._self_delegate = _PolicyAgent(power, self._policy)

    def _plan_orders(self, state, round_index: int) -> List[Order]:
        candidate_map = self._policy._build_candidate_order_map(state, self.power)  # type: ignore[attr-defined]
        combos = self._policy._enumerate_combos(candidate_map)  # type: ignore[attr-defined]
        if not combos:
            return []

        best_score = float("-inf")
        best_orders: Sequence[Order] = combos[0]
        for combo in combos[: self._max_candidates]:
            score = self._score_candidate(state, combo)
            if score > best_score:
                best_score = score
                best_orders = combo
        return list(best_orders)

    def _score_candidate(self, state, candidate_orders: Iterable[Order]) -> float:
        agents = self._build_rollout_agents(state)
        return save(
            initial_state=state,
            focal_power=self.power,
            candidate_orders=candidate_orders,
            agents=agents,
            n_rollouts=self._n_rollouts,
            horizon=self._horizon,
            value_fn=self._value_fn,
            value_kwargs=self._value_kwargs,
        )

    def _build_rollout_agents(self, state) -> Dict[Power, Agent]:
        agents: Dict[Power, Agent] = {}
        for p in state.powers:
            agents[p] = self._self_delegate if p == self.power else self._opponent_factory(p)
        return agents


__all__ = ["SaveBestResponseAgent"]
