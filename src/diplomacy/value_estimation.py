from __future__ import annotations
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .state import GameState
from .types import Order, Power
from .agents.base import Agent
from .deepmind.build_observation import build_observation
from .deepmind.actions import legal_actions_from_state, decode_actions_to_orders
from .timing import timer
from policytraining.run_sl import make_sl_policy

ValueFn = Callable[[GameState, Power], float]
BatchValueFn = Callable[[Sequence[GameState], Power], Sequence[float]]
OrderMap = Mapping[Power, Iterable[Order]]


@lru_cache(maxsize=1)
def _load_sl_policy(weights_path: str, rng_seed: int, temperature: float = 0.1):
    return make_sl_policy(weights_path, rng_seed=rng_seed)


def sl_state_value(
    state: GameState,
    power: Power,
    *,
    policy=None,
    weights_path: Optional[str] = None,
    rng_seed: int = 0,
) -> float:
    """DeepMind SL value head for `power` at `state` (matches DeepMindSlAgent)."""
    if policy is None:
        if weights_path is None:
            raise ValueError("weights_path required when policy is not provided")
        policy = _load_sl_policy(weights_path, rng_seed=rng_seed)

    observation = build_observation(state, last_actions=[])
    legal_actions = legal_actions_from_state(state)
    if not legal_actions:
        return 0.0

    powers: List[Power] = sorted(state.powers, key=str)
    slots_list: Sequence[int] = list(range(len(powers)))
    with timer("model_inference_value"):
        _, info = policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=list(legal_actions),
        )

    values = info.get("values") if isinstance(info, dict) else None
    if values is None:
        return 0.0
    try:
        return float(values[powers.index(power)])
    except (ValueError, IndexError, TypeError):
        return 0.0


def batch_sl_state_value(
    states: Sequence[GameState],
    power: Power,
    *,
    policy=None,
    weights_path: Optional[str] = None,
    rng_seed: int = 0,
) -> Sequence[float]:
    """Batch-friendly wrapper around ``sl_state_value`` (falls back to per-state calls)."""
    return [
        sl_state_value(
            state,
            power,
            policy=policy,
            weights_path=weights_path,
            rng_seed=rng_seed,
        )
        for state in states
    ]


def heuristic_state_value(
    state: GameState,
    power: Power,
    *,
    center_weight: float = 1.0,
    unit_weight: float = 0.25,
    threatened_weight: float = 0.5,
) -> float:
    """Lightweight heuristic value function for faster rollouts.

    Scores the board based on controlled supply centers, unit count,
    and threatened owned centers. The weights can be tuned per experiment.
    """
    centers_controlled = sum(
        1 for controller in state.supply_center_control.values() if controller == power
    )
    unit_count = sum(1 for unit in state.units.values() if unit.power == power)
    threatened = state.centers_threatened(power)
    return (
        center_weight * centers_controlled
        + unit_weight * unit_count
        - threatened_weight * threatened
    )


def _copy_state(state: GameState) -> GameState:
    # Replace with your preferred cloning method if different.
    return state.copy()


def rollout_value(
    initial_state: GameState,
    focal_power: Power,
    fixed_orders_by_power: OrderMap,
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn: ValueFn = sl_state_value,
    value_kwargs: Optional[dict] = None,
    batch_value_fn: Optional[BatchValueFn] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Average value over rollouts with some powers' orders fixed in the first round."""
    from .simulation import run_rounds_with_agents  # local import to avoid circulars
    rng = rng or random.Random()
    value_kwargs = value_kwargs or {}
    total = 0.0
    final_states: List[GameState] = []

    for _ in range(n_rollouts):
        state = _copy_state(initial_state)
        rollout_agents: Dict[Power, Agent] = {}
        for power, agent in agents.items():
            fixed = fixed_orders_by_power.get(power)
            if fixed is None:
                rollout_agents[power] = agent
                continue

            class FixedThenAgent(Agent):
                def __init__(self, inner: Agent, fixed_orders: Iterable[Order]):
                    super().__init__(power=inner.power)
                    self.inner = inner
                    self.fixed = list(fixed_orders)
                    self.used = False

                def _plan_orders(self, state: GameState, round_index: int) -> List[Order]:
                    if not self.used:
                        self.used = True
                        return self.fixed
                    return self.inner._plan_orders(state, round_index)

            rollout_agents[power] = FixedThenAgent(agent, fixed)

        states, _, _ = run_rounds_with_agents(
            state,
            rollout_agents,
            horizon,
            stop_on_winner=False,
        )
        final_states.append(states[-1])

    if batch_value_fn is not None:
        values = batch_value_fn(final_states, focal_power)
        total = float(sum(values))
    else:
        for final_state in final_states:
            total += value_fn(final_state, focal_power, **value_kwargs)

    return total / max(1, n_rollouts)


def save(
    initial_state: GameState,
    focal_power: Power,
    candidate_orders: Iterable[Order],
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn: ValueFn = sl_state_value,
    value_kwargs: Optional[dict] = None,
    batch_value_fn: Optional[BatchValueFn] = None,
    rng: Optional[random.Random] = None,
) -> float:
    return rollout_value(
        initial_state,
        focal_power,
        {focal_power: candidate_orders},
        agents,
        n_rollouts=n_rollouts,
        horizon=horizon,
        value_fn=value_fn,
        value_kwargs=value_kwargs,
        batch_value_fn=batch_value_fn,
        rng=rng,
    )


def stave(
    initial_state: GameState,
    focal_power: Power,
    partner_power: Power,
    focal_orders: Iterable[Order],
    partner_orders: Iterable[Order],
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn: ValueFn = sl_state_value,
    value_kwargs: Optional[dict] = None,
    batch_value_fn: Optional[BatchValueFn] = None,
    rng: Optional[random.Random] = None,
) -> float:
    return rollout_value(
        initial_state,
        focal_power,
        {focal_power: focal_orders, partner_power: partner_orders},
        agents,
        n_rollouts=n_rollouts,
        horizon=horizon,
        value_fn=value_fn,
        value_kwargs=value_kwargs,
        batch_value_fn=batch_value_fn,
        rng=rng,
    )


@dataclass(frozen=True)
class Contract:
    proposer: Power
    responder: Power
    proposer_orders: Iterable[Order]
    responder_orders: Iterable[Order]


def sve(
    initial_state: GameState,
    focal_power: Power,
    contract: Contract,
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn: ValueFn = sl_state_value,
    value_kwargs: Optional[dict] = None,
    batch_value_fn: Optional[BatchValueFn] = None,
    rng: Optional[random.Random] = None,
) -> float:
    fixed = {
        contract.proposer: contract.proposer_orders,
        contract.responder: contract.responder_orders,
    }
    return rollout_value(
        initial_state,
        focal_power,
        fixed,
        agents,
        n_rollouts=n_rollouts,
        horizon=horizon,
        value_fn=value_fn,
        value_kwargs=value_kwargs,
        batch_value_fn=batch_value_fn,
        rng=rng,
    )


__all__ = [
    "BatchValueFn",
    "batch_sl_state_value",
    "sl_state_value",
    "heuristic_state_value",
    "rollout_value",
    "save",
    "stave",
    "sve",
    "Contract",
    "ValueFn",
]
