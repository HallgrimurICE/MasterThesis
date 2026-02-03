from __future__ import annotations

from typing import Dict, List, Tuple

from .adjudication import Adjudicator
from .agents.base import Agent
from .state import GameState
from .types import Order, Power, UnitType
from .timing import get_timing_recorder, timer


def     run_rounds_with_agents(
    initial_state: GameState,
    agents: Dict[Power, Agent],
    rounds: int,
    *,
    title_prefix: str = "After Round {round}",
    stop_on_winner: bool = False,
) -> Tuple[List[GameState], List[str], List[List[Order]]]:
    """Execute a number of movement rounds using programmable agents."""

    state = initial_state
    states = [state]
    titles = ["Initial State"]
    orders_history: List[List[Order]] = []

    def collect_build_choices(build_state: GameState) -> Dict[Power, List[Tuple[str, UnitType]]]:
        selections: Dict[Power, List[Tuple[str, UnitType]]] = {}
        for build_power, count in build_state.pending_builds.items():
            if count <= 0:
                continue
            agent = agents.get(build_power)
            if agent is None:
                continue
            choices = agent.plan_builds(build_state, count)
            if choices:
                selections[build_power] = choices
        return selections

    movement_round = 0
    recorder = get_timing_recorder()
    while movement_round < rounds and (not stop_on_winner or state.winner is None):
        if state.phase.name.endswith("RETREAT"):
            if recorder is not None:
                recorder.start_round(movement_round)
            print(f"[Round {movement_round}] (Retreat phase)")
            retreat_orders: List[Order] = []
            with timer("planning"):
                for power, agent in agents.items():
                    if not any(u.power == power for u in state.pending_retreats.values()):
                        continue
                    retreat_orders.extend(agent.issue_orders(state))
            with timer("adjudication"):
                state, resolution = Adjudicator(state).resolve(
                    retreat_orders,
                    build_callback=collect_build_choices,
                )
            states.append(state)
            title = f"{title_prefix.format(round=movement_round)} (Retreat)"
            if stop_on_winner and resolution.winner is not None:
                title += f" – Winner: {resolution.winner}"
            if resolution.auto_disbands:
                summary = ", ".join(
                    f"{power}: {', '.join(sorted(provinces))}"
                    for power, provinces in resolution.auto_disbands.items()
                )
                title += f" [Disbands: {summary}]"
            if resolution.auto_builds:
                summary = ", ".join(
                    f"{power}: {', '.join(sorted(provinces))}"
                    for power, provinces in resolution.auto_builds.items()
                )
                title += f" [Builds: {summary}]"
            titles.append(title)
            orders_history.append(retreat_orders)
            if stop_on_winner and resolution.winner is not None:
                if recorder is not None:
                    recorder.end_round()
                break
            if recorder is not None:
                recorder.end_round()
            continue

        movement_round += 1
        if recorder is not None:
            recorder.start_round(movement_round)
        # if movement_round % 1 == 0:
            # print(f"[Round {movement_round}]")
    
        round_orders: List[Order] = []
        with timer("planning"):
            for power, agent in agents.items():
                if power not in state.powers:
                    continue
                agent_orders = agent.issue_orders(state)
                round_orders.extend(agent_orders)

        orders_history.append(list(round_orders))
        with timer("adjudication"):
            state, resolution = Adjudicator(state).resolve(
                round_orders,
                build_callback=collect_build_choices,
            )
        states.append(state)
        title = title_prefix.format(round=movement_round)
        if stop_on_winner and resolution.winner is not None:
            title += f" – Winner: {resolution.winner}"
        if resolution.auto_disbands:
            summary = ", ".join(
                f"{power}: {', '.join(sorted(provinces))}"
                for power, provinces in resolution.auto_disbands.items()
            )
            title += f" [Disbands: {summary}]"
        if resolution.auto_builds:
            summary = ", ".join(
                f"{power}: {', '.join(sorted(provinces))}"
                for power, provinces in resolution.auto_builds.items()
            )
            title += f" [Builds: {summary}]"
        titles.append(title)
        if stop_on_winner and resolution.winner is not None:
            if recorder is not None:
                recorder.end_round()
            break
        if recorder is not None:
            recorder.end_round()

    return states, titles, orders_history


__all__ = ["run_rounds_with_agents"]
