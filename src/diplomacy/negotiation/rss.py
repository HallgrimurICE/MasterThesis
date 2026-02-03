"""Round-Simultaneous Signaling (RSS) with Peace contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..state import GameState
from ..types import Power
from .contracts import Contract
from .peace import build_peace_contract
from .simulation import BatchValueFn, PolicyFn, StepFn, ValueFn, estimate_expected_values


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
    tom_depth: int = 1,
    batch_value_fn: Optional[BatchValueFn] = None,
    value_cache: Optional[MutableMapping[Tuple[Tuple[Any, ...], Tuple[Tuple[Power, Tuple[int, ...]], ...]], Mapping[Power, float]]] = None,
    state_signature: Optional[Tuple[Any, ...]] = None,
) -> Set[Power]:
    """Return the set of powers to whom ``power`` proposes Peace.

    tom_depth=0 proposes peace with everyone (no value reasoning). tom_depth=1
    uses only the focal power's improvement. tom_depth>=2 also requires the
    counterpart's expected value to improve under the deal.
    """

    baseline_values = _cached_expected_values(
        state,
        powers=powers,
        policy_fns=policy_fns,
        value_fn=value_fn,
        step_fn=step_fn,
        legal_actions=legal_actions,
        rollouts=rollouts,
        batch_value_fn=batch_value_fn,
        value_cache=value_cache,
        state_signature=state_signature,
        restricted_actions=None,
    )
    baseline = baseline_values.get(power, 0.0)

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
        deal_values = _cached_expected_values(
            state,
            powers=powers,
            policy_fns=policy_fns,
            value_fn=value_fn,
            step_fn=step_fn,
            legal_actions=legal_actions,
            restricted_actions=restrictions,
            rollouts=rollouts,
            batch_value_fn=batch_value_fn,
            value_cache=value_cache,
            state_signature=state_signature,
        )
        if tom_depth <= 0:
            proposals.add(other)
            continue
        deal_value = deal_values.get(power, baseline)
        if deal_value <= baseline:
            continue
        if tom_depth >= 2:
            other_baseline = baseline_values.get(other, 0.0)
            other_deal_value = deal_values.get(other, other_baseline)
            if other_deal_value <= other_baseline:
                continue
        proposals.add(other)
    return proposals


def _cached_expected_values(
    state: GameState,
    *,
    powers: Sequence[Power],
    policy_fns: Mapping[Power, PolicyFn],
    value_fn: ValueFn,
    step_fn: StepFn,
    legal_actions: Mapping[Power, Sequence[int]],
    restricted_actions: Optional[Mapping[Power, Sequence[int]]],
    rollouts: int,
    batch_value_fn: Optional[BatchValueFn],
    value_cache: Optional[MutableMapping[Tuple[Tuple[Any, ...], Tuple[Tuple[Power, Tuple[int, ...]], ...]], Mapping[Power, float]]],
    state_signature: Optional[Tuple[Any, ...]],
) -> Mapping[Power, float]:
    signature = _cache_signature(state_signature, restricted_actions)
    if value_cache is not None and signature in value_cache:
        return value_cache[signature]

    values = estimate_expected_values(
        state,
        target_powers=powers,
        policy_fns=policy_fns,
        value_fn=value_fn,
        step_fn=step_fn,
        legal_actions=legal_actions,
        restricted_actions=restricted_actions,
        rollouts=rollouts,
        batch_value_fn=batch_value_fn,
    )
    if value_cache is not None:
        value_cache[signature] = values
    return values


def _cache_signature(
    state_signature: Optional[Tuple[Any, ...]],
    restricted_actions: Optional[Mapping[Power, Sequence[int]]],
) -> Tuple[Tuple[Any, ...], Tuple[Tuple[Power, Tuple[int, ...]], ...]]:
    restriction_signature: Tuple[Tuple[Power, Tuple[int, ...]], ...]
    if not restricted_actions:
        restriction_signature = tuple()
    else:
        restriction_signature = tuple(
            sorted(
                (power, tuple(sorted(int(action) for action in actions)))
                for power, actions in restricted_actions.items()
            )
        )
    return (state_signature or tuple(), restriction_signature)


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
