from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..types import Order, Power
from ..agents import Agent
from ..value_estimation import (
    save,
    stave,
    sve,
    sl_state_value,
    Contract as ValueContract,
)

def rank_actions_with_save(
    state,
    focal_power: Power,
    candidate_orders_list: Sequence[Iterable[Order]],
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn=sl_state_value,
    value_kwargs: Optional[dict] = None,
) -> List[Tuple[List[Order], float]]:
    """Score multiple candidate order sets for one power; returns sorted best→worst."""
    scores: List[Tuple[List[Order], float]] = []
    for orders in candidate_orders_list:
        score = save(
            initial_state=state,
            focal_power=focal_power,
            candidate_orders=orders,
            agents=agents,
            n_rollouts=n_rollouts,
            horizon=horizon,
            value_fn=value_fn,
            value_kwargs=value_kwargs,
        )
        scores.append((list(orders), score))
    return sorted(scores, key=lambda item: item[1], reverse=True)

def best_action_with_save(
    state,
    focal_power: Power,
    candidate_orders_list: Sequence[Iterable[Order]],
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn=sl_state_value,
    value_kwargs: Optional[dict] = None,
) -> Tuple[List[Order], float]:
    """Argmax wrapper around rank_actions_with_save."""
    ranked = rank_actions_with_save(
        state,
        focal_power,
        candidate_orders_list,
        agents,
        n_rollouts=n_rollouts,
        horizon=horizon,
        value_fn=value_fn,
        value_kwargs=value_kwargs,
    )
    return ranked[0]

def score_contract(
    state,
    contract: ValueContract,
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn=sl_state_value,
    value_kwargs: Optional[dict] = None,
) -> Dict[Power, float]:
    """Evaluate a bilateral contract from both parties' perspectives."""
    proposer = contract.proposer
    responder = contract.responder
    return {
        proposer: sve(
            initial_state=state,
            focal_power=proposer,
            contract=contract,
            agents=agents,
            n_rollouts=n_rollouts,
            horizon=horizon,
            value_fn=value_fn,
            value_kwargs=value_kwargs,
        ),
        responder: sve(
            initial_state=state,
            focal_power=responder,
            contract=contract,
            agents=agents,
            n_rollouts=n_rollouts,
            horizon=horizon,
            value_fn=value_fn,
            value_kwargs=value_kwargs,
        ),
    }

def rank_joint_with_stave(
    state,
    focal_power: Power,
    partner_power: Power,
    candidate_pairs: Sequence[Tuple[Iterable[Order], Iterable[Order]]],
    agents: Mapping[Power, Agent],
    *,
    n_rollouts: int = 4,
    horizon: int = 1,
    value_fn=sl_state_value,
    value_kwargs: Optional[dict] = None,
) -> List[Tuple[Tuple[List[Order], List[Order]], float]]:
    """Score joint (focal, partner) order pairs for the focal power; returns sorted best→worst."""
    scored: List[Tuple[Tuple[List[Order], List[Order]], float]] = []
    for my_orders, partner_orders in candidate_pairs:
        score = stave(
            initial_state=state,
            focal_power=focal_power,
            partner_power=partner_power,
            focal_orders=my_orders,
            partner_orders=partner_orders,
            agents=agents,
            n_rollouts=n_rollouts,
            horizon=horizon,
            value_fn=value_fn,
            value_kwargs=value_kwargs,
        )
        scored.append(((list(my_orders), list(partner_orders)), score))
    return sorted(scored, key=lambda item: item[1], reverse=True)
