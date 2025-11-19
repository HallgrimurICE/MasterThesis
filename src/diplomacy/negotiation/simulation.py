"""Monte-Carlo helpers for evaluating contracts under RSS."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

from ..state import GameState
from ..types import Power

PolicyFn = Callable[
    [GameState, Power, Mapping[Power, Sequence[int]], Optional[Mapping[Power, Sequence[int]]]],
    Sequence[int],
]
ValueFn = Callable[[GameState, Power], float]
StepFn = Callable[[GameState, Mapping[Power, Sequence[int]]], GameState]


def estimate_expected_value(
    state: GameState,
    target_power: Power,
    policy_fns: Mapping[Power, PolicyFn],
    value_fn: ValueFn,
    step_fn: StepFn,
    legal_actions: Mapping[Power, Sequence[int]],
    *,
    restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    rollouts: int = 4,
) -> float:
    """Approximate ``E[V_i(next_state)]`` for ``target_power``.

    Each rollout samples a joint action profile using the provided ``policy_fns``
    (optionally restricted by a contract), advances the environment once via
    ``step_fn``, and queries the shared value network on the resulting state.
    """

    if not policy_fns:
        return 0.0

    total = 0.0
    for _ in range(max(1, rollouts)):
        joint_actions: Dict[Power, Sequence[int]] = {}
        for power, policy_fn in policy_fns.items():
            joint_actions[power] = policy_fn(state, power, legal_actions, restricted_actions)
        next_state = step_fn(state, joint_actions)
        total += value_fn(next_state, target_power)
    return total / max(1, rollouts)


__all__ = ["estimate_expected_value"]
