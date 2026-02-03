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
BatchValueFn = Callable[[Sequence[GameState], Power], Sequence[float]]
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
    batch_value_fn: Optional[BatchValueFn] = None,
) -> float:
    """Approximate ``E[V_i(next_state)]`` for ``target_power``."""

    values = estimate_expected_values(
        state,
        target_powers=[target_power],
        policy_fns=policy_fns,
        value_fn=value_fn,
        step_fn=step_fn,
        legal_actions=legal_actions,
        restricted_actions=restricted_actions,
        rollouts=rollouts,
        batch_value_fn=batch_value_fn,
    )
    return values.get(target_power, 0.0)


def estimate_expected_values(
    state: GameState,
    *,
    target_powers: Sequence[Power],
    policy_fns: Mapping[Power, PolicyFn],
    value_fn: ValueFn,
    step_fn: StepFn,
    legal_actions: Mapping[Power, Sequence[int]],
    restricted_actions: Optional[Mapping[Power, Sequence[int]]] = None,
    rollouts: int = 4,
    batch_value_fn: Optional[BatchValueFn] = None,
) -> Mapping[Power, float]:
    """Approximate expected values for all ``target_powers`` in one sweep."""

    if not policy_fns or not target_powers:
        return {power: 0.0 for power in target_powers}

    totals: Dict[Power, float] = {power: 0.0 for power in target_powers}
    next_states: list[GameState] = []
    for _ in range(max(1, rollouts)):
        joint_actions: Dict[Power, Sequence[int]] = {}
        for power, policy_fn in policy_fns.items():
            joint_actions[power] = policy_fn(state, power, legal_actions, restricted_actions)
        next_state = step_fn(state, joint_actions)
        next_states.append(next_state)

    if batch_value_fn is not None:
        for power in target_powers:
            totals[power] = float(sum(batch_value_fn(next_states, power)))
    else:
        for next_state in next_states:
            for power in target_powers:
                totals[power] += value_fn(next_state, power)

    norm = float(max(1, rollouts))
    return {power: total / norm for power, total in totals.items()}


__all__ = ["BatchValueFn", "estimate_expected_value", "estimate_expected_values"]
