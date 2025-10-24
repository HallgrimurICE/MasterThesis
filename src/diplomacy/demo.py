from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .adjudication import Adjudicator, Resolution
from .agents import Agent, RandomAgent
from .maps import cooperative_attack_initial_state, demo_state_mesh
from .orders import hold, move, support_hold, support_move
from .simulation import run_rounds_with_agents
from .state import GameState
from .types import Order, Power, Unit, describe_order
from .viz.mesh import interactive_visualize_state_mesh


def simulate_two_power_cooperation() -> Dict[str, Tuple[GameState, Resolution, List[Order]]]:
    """Compare outcomes with and without coordinated support."""

    scenarios: Dict[str, Tuple[GameState, Resolution, List[Order]]] = {}

    def run_case(order_builder: Callable[[GameState], List[Order]], label: str) -> None:
        state = cooperative_attack_initial_state()
        orders = order_builder(state)
        next_state, resolution = Adjudicator(state).resolve(orders)
        scenarios[label] = (next_state, resolution, orders)

    def solo_attack(state: GameState) -> List[Order]:
        attacker = state.units["A"]
        ally = state.units["B"]
        defender = state.units["C"]
        return [
            move(attacker, "C"),
            hold(ally),
            hold(defender),
        ]

    def supported_attack(state: GameState) -> List[Order]:
        attacker = state.units["A"]
        supporter = state.units["B"]
        defender = state.units["C"]
        return [
            move(attacker, "C"),
            support_move(supporter, "A", "C"),
            hold(defender),
        ]

    run_case(solo_attack, "solo_attack")
    run_case(supported_attack, "supported_attack")

    return scenarios


def _format_orders_with_actions(orders: Iterable[Order]) -> List[str]:
    order_list = list(orders)
    if not order_list:
        return ["  (no orders issued)"]

    order_lines = [f"  * {order}" for order in order_list]
    descriptions = [describe_order(order) for order in order_list]
    max_line = max(len(text) for text in order_lines)
    header_order = "Order"
    header_action = "Action"
    formatted: List[str] = []
    formatted.append(f"  {header_order}".ljust(max_line) + f" | {header_action}")
    formatted.append(f"{'-' * max_line}-+-{'-' * len(header_action)}")
    for line, description in zip(order_lines, descriptions):
        formatted.append(f"{line.ljust(max_line)} | {description}")
    return formatted


def print_two_power_cooperation_report() -> None:
    initial_state = cooperative_attack_initial_state()
    print("=== Cooperative attack scenario ===")
    print("Initial unit placement:")
    for loc in sorted(initial_state.units):
        unit = initial_state.units[loc]
        sc_flag = " (SC)" if initial_state.board[loc].is_supply_center else ""
        print(f"  - {unit.power} unit in {loc}{sc_flag}")

    outcomes = simulate_two_power_cooperation()
    for label in ("solo_attack", "supported_attack"):
        next_state, resolution, orders = outcomes[label]
        print(f"\nScenario: {label.replace('_', ' ').title()}")
        print("Orders issued:")
        for line in _format_orders_with_actions(orders):
            print(line)
        succeeded = sorted(str(o) for o in resolution.succeeded)
        failed = sorted(str(o) for o in resolution.failed)
        dislodged = sorted(resolution.dislodged)
        print("Succeeded orders:")
        for text in succeeded:
            print(f"    {text}")
        print("Failed orders:")
        for text in failed:
            print(f"    {text}")
        print(f"Dislodged provinces: {dislodged if dislodged else 'None'}")
        occupying = {loc: unit.power for loc, unit in sorted(next_state.units.items())}
        print("Post-resolution occupants:")
        for loc, power in occupying.items():
            print(f"  - {loc}: {power}")


def demo_run_mesh_with_random_orders(rounds: int = 3):
    state = demo_state_mesh()
    states = [state]
    titles = ["Initial 5x3 Mesh Map"]

    toward = {"1": "7", "5": "9", "11": "12", "15": "14", "8": "8"}

    for r in range(1, rounds + 1):
        orders: List[Order] = []
        for loc, unit in list(state.units.items()):
            if unit.power == Power("Red"):
                orders.append(hold(unit))
                continue
            dest = toward.get(loc)
            if dest and dest in state.legal_moves_from(loc):
                orders.append(move(unit, dest))
            else:
                orders.append(hold(unit))

        print(f"\nRound {r} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)
        state, _ = Adjudicator(state).resolve(orders)
        states.append(state)
        titles.append(f"After Round {r} – 5x3 Mesh Map")

    interactive_visualize_state_mesh(states, titles)


def demo_run_mesh_with_random_agents(
    rounds: int = 500,
    *,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
) -> None:
    state = demo_state_mesh()
    base_rng = random.Random(seed)

    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        agent_seed = base_rng.randint(0, 2**32 - 1)
        agents[power] = RandomAgent(
            power,
            hold_probability=hold_probability,
            rng=random.Random(agent_seed),
        )

    states, titles, orders_history = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="After Round {round} – Random Agents on 5x3 Mesh",
        stop_on_winner=True,
    )

    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)

    winner = states[-1].winner
    if winner is not None:
        print(f"\nWinner detected: {winner} controls a majority of supply centers.")
    else:
        print("\nNo winner within the configured round limit.")

    interactive_visualize_state_mesh(states, titles)


__all__ = [
    "simulate_two_power_cooperation",
    "print_two_power_cooperation_report",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
]


if __name__ == "__main__":
    print("Running demo_run_mesh_with_random_agents() with default parameters...")
    demo_run_mesh_with_random_agents()
