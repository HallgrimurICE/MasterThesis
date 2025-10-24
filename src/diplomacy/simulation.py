from __future__ import annotations

from typing import Dict, List, Tuple

from .adjudication import Adjudicator
from .agents import Agent
from .state import GameState
from .types import Order, Power


def run_rounds_with_agents(
    initial_state: GameState,
    agents: Dict[Power, Agent],
    rounds: int,
    *,
    title_prefix: str = "After Round {round}",
) -> Tuple[List[GameState], List[str], List[List[Order]]]:
    """Execute a number of movement rounds using programmable agents."""

    state = initial_state
    states = [state]
    titles = ["Initial State"]
    orders_history: List[List[Order]] = []

    movement_round = 0
    while movement_round < rounds:
        if state.phase.name.endswith("RETREAT"):
            retreat_orders: List[Order] = []
            for power, agent in agents.items():
                if not any(u.power == power for u in state.pending_retreats.values()):
                    continue
                retreat_orders.extend(agent.issue_orders(state))
            state, _ = Adjudicator(state).resolve(retreat_orders)
            states.append(state)
            titles.append(f"{title_prefix.format(round=movement_round)} (Retreat)")
            orders_history.append(retreat_orders)
            continue

        movement_round += 1
        round_orders: List[Order] = []
        for power, agent in agents.items():
            if power not in state.powers:
                continue
            agent_orders = agent.issue_orders(state)
            round_orders.extend(agent_orders)

        orders_history.append(list(round_orders))
        state, _ = Adjudicator(state).resolve(round_orders)
        states.append(state)
        titles.append(title_prefix.format(round=movement_round))

    return states, titles, orders_history


__all__ = ["run_rounds_with_agents"]
