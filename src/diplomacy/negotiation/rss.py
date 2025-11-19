"""Round-Simultaneous Signaling (RSS) with Peace contracts."""

from __future__ import annotations

from typing import Mapping, Sequence, Set

from ..state import GameState
from ..types import Power
from .contracts import Contract
from .peace import build_peace_contract
from .simulation import PolicyFn, StepFn, ValueFn, estimate_expected_value


def run_rss_for_power(
    state: GameState,
    power: Power,
    powers: Sequence[Power],
    legal_actions: Mapping[Power, Sequence[int]],
    policy_fns: Mapping[Power, PolicyFn],
    value_fn: ValueFn,
    step_fn: StepFn,
    *,
    rollouts: int = 4,
) -> Set[Power]:
    """Return the set of powers to whom ``power`` proposes Peace."""

    baseline = estimate_expected_value(
        state,
        target_power=power,
        policy_fns=policy_fns,
        value_fn=value_fn,
        step_fn=step_fn,
        legal_actions=legal_actions,
        rollouts=rollouts,
    )

    proposals: Set[Power] = set()
    for other in powers:
        if other == power:
            continue
        contract = build_peace_contract(
            state,
            power,
            other,
            legal_actions[power],
            legal_actions[other],
        )
        restrictions = {
            power: tuple(contract.allowed_i),
            other: tuple(contract.allowed_j),
        }
        deal_value = estimate_expected_value(
            state,
            target_power=power,
            policy_fns=policy_fns,
            value_fn=value_fn,
            step_fn=step_fn,
            legal_actions=legal_actions,
            restricted_actions=restrictions,
            rollouts=rollouts,
        )
        if deal_value > baseline:
            proposals.add(other)
    return proposals


def compute_active_contracts(
    state: GameState,
    powers: Sequence[Power],
    legal_actions: Mapping[Power, Sequence[int]],
    proposals: Mapping[Power, Set[Power]],
) -> Sequence[Contract]:
    """Return the mutually agreed Peace contracts derived from RSS proposals."""

    active: list[Contract] = []
    for idx, power in enumerate(powers):
        for other in powers[idx + 1 :]:
            if other not in proposals.get(power, set()):
                continue
            if power not in proposals.get(other, set()):
                continue
            active.append(
                build_peace_contract(state, power, other, legal_actions[power], legal_actions[other])
            )
    return active


__all__ = ["run_rss_for_power", "compute_active_contracts"]
