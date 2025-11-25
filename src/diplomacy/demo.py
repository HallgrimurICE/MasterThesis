from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .adjudication import Adjudicator, Resolution
from .agents import (
    Agent,
    ObservationBestResponseAgent,
    RandomAgent,
    SampledBestResponsePolicy,
)
from .maps import (
    cooperative_attack_initial_state,
    demo_state_mesh,
    fleet_coast_demo_state,
    standard_initial_state,
)
from .agents.sl_agent import DeepMindSlAgent, BaselineNegotiatorAgent
from .orders import hold, move, support_hold, support_move
from .simulation import run_rounds_with_agents
from .state import GameState
from .types import Order, Power, Unit, describe_order
from .viz.mesh import interactive_visualize_state_mesh, visualize_state


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


def simulate_fleet_coast_movements() -> Dict[str, Tuple[GameState, Resolution, List[Order]]]:
    """Demonstrate movement validation for armies vs fleets on mixed terrain."""

    scenarios: Dict[str, Tuple[GameState, Resolution, List[Order]]] = {}

    def run_case(order_builder: Callable[[GameState], List[Order]], label: str) -> None:
        state = fleet_coast_demo_state()
        orders = order_builder(state)
        next_state, resolution = Adjudicator(state).resolve(orders)
        scenarios[label] = (next_state, resolution, orders)

    def illegal_orders(state: GameState) -> List[Order]:
        units = state.units
        return [
            move(units["Highlands"], "AzureSea"),  # army attempting to enter sea
            move(units["AlbionBay"], "Highlands"),  # fleet attempting to enter land
            move(units["RedKeep"], "AzureSea"),  # army attempting to sail
            move(units["OpenOcean"], "RedKeep"),  # fleet attempting to landlocked province
        ]

    def terrain_compliant_orders(state: GameState) -> List[Order]:
        units = state.units
        return [
            move(units["Highlands"], "Cliffhaven"),
            move(units["AlbionBay"], "AzureSea"),
            hold(units["RedKeep"]),
            hold(units["OpenOcean"]),
        ]

    run_case(illegal_orders, "illegal_orders")
    run_case(terrain_compliant_orders, "terrain_compliant_orders")

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


def print_fleet_coast_demo_report() -> None:
    initial_state = fleet_coast_demo_state()
    print("=== Fleet and Coast Movement Scenario ===")
    print("Initial terrain and occupants:")
    for loc in sorted(initial_state.board):
        province = initial_state.board[loc]
        unit = initial_state.units.get(loc)
        terrain = province.province_type.name.title()
        sc_flag = " (SC)" if province.is_supply_center else ""
        if unit:
            unit_desc = f"{unit.power} {unit.unit_type.name.title()}"
        else:
            unit_desc = "Unoccupied"
        print(f"  - {loc}: {terrain}{sc_flag} -> {unit_desc}")

    print("\nLegal moves from initial state:")
    for loc, unit in sorted(initial_state.units.items()):
        moves = sorted(initial_state.legal_moves_from(loc))
        move_text = ", ".join(moves) if moves else "(none)"
        print(f"  * {unit.power} {unit.unit_type.name.title()} in {loc}: {move_text}")

    outcomes = simulate_fleet_coast_movements()
    for label in ("illegal_orders", "terrain_compliant_orders"):
        next_state, resolution, orders = outcomes[label]
        print(f"\nScenario: {label.replace('_', ' ').title()}")
        print("Orders issued:")
        for line in _format_orders_with_actions(orders):
            print(line)
        print("Succeeded orders:")
        for text in sorted(str(o) for o in resolution.succeeded):
            print(f"    {text}")
        print("Failed orders:")
        for text in sorted(str(o) for o in resolution.failed):
            print(f"    {text}")
        print("Post-resolution occupants:")
        for loc, unit in sorted(next_state.units.items()):
            print(f"  - {loc}: {unit.power} {unit.unit_type.name.title()}")


def print_standard_board_demo() -> None:
    """Quick inspection helper for the emerging standard Diplomacy board."""

    state = standard_initial_state()
    print("=== Standard Map (Work in Progress) ===")
    print(f"Provinces defined: {len(state.board)}")
    print(f"Units placed: {len(state.units)}")
    for name in sorted(state.board):
        province = state.board[name]
        terrain = province.province_type.name.title()
        sc_flag = "Yes" if province.is_supply_center else "No"
        home = str(province.home_power) if province.home_power is not None else "-"
        neighbors = ", ".join(sorted(province.neighbors)) if province.neighbors else "(none yet)"
        print(f"\n{name}")
        print(f"  Terrain: {terrain}")
        print(f"  Supply Center: {sc_flag}")
        print(f"  Home Power: {home}")
        print(f"  Neighbors: {neighbors}")

    if not state.units:
        print("\n(No starting units added yet.)")


def visualize_fleet_coast_demo() -> None:
    """Interactive visualization for the fleet/coast terrain demo."""

    scenarios = simulate_fleet_coast_movements()
    initial_state = fleet_coast_demo_state()

    states: List[GameState] = [initial_state]
    titles: List[str] = ["Initial Fleet/Coast Demo Map"]

    illegal_state, _, _ = scenarios["illegal_orders"]
    states.append(illegal_state)
    titles.append("After Illegal Orders (all rejected)")

    legal_state, _, _ = scenarios["terrain_compliant_orders"]
    states.append(legal_state)
    titles.append("After Terrain-Compliant Orders")

    interactive_visualize_state_mesh(states, titles)


def visualize_standard_board() -> None:
    """Visualize the current standard Diplomacy map (requires matplotlib/networkx)."""

    state = standard_initial_state()
    visualize_state(state, title="Standard Diplomacy Map (Work in Progress)")


def interactive_visualize_standard_board() -> None:
    """Interactive view of the current standard Diplomacy map."""

    state = standard_initial_state()
    interactive_visualize_state_mesh(
        [state],
        ["Standard Diplomacy Map (Work in Progress)"],
    )


def run_standard_board_with_random_england(
    rounds: int = 5,
    *,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
) -> None:
    """Run a short simulation with England driven by a random agent on the standard board."""

    state = standard_initial_state()
    agent_seed = random.Random(seed).randint(0, 2**32 - 1)
    england_agent = RandomAgent(
        Power("England"),
        hold_probability=hold_probability,
        rng=random.Random(agent_seed),
    )

    agents: Dict[Power, Agent] = {Power("England"): england_agent}

    states, titles, orders_history = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="Standard Board After Round {round}",
        stop_on_winner=False,
    )

    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders (England):")
        for line in _format_orders_with_actions(orders):
            print(line)

    interactive_visualize_state_mesh(states, titles)


def run_standard_board_with_random_agents(
    rounds: int = 1000,
    visualize: bool = False,
    *,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    stop_on_winner: bool = True,
    policy_power: Optional[Power] = Power("France"),
) -> None:
    """Run the standard board with one policy-driven power and the rest random.

    Args:
        rounds: Number of movement rounds to simulate.
        visualize: Whether to open the visualization mesh at the end.
        seed: Optional PRNG seed for reproducibility.
        hold_probability: Probability a random agent holds instead of acting.
        stop_on_winner: Stop early when a winner is detected.
        policy_power: Power to control via ``ObservationBestResponseAgent``;
            set to ``None`` to make every power random.
    """

    state = standard_initial_state()
    base_rng = random.Random(seed)

    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        agent_seed = base_rng.randint(0, 2**32 - 1)
        if policy_power is not None and power == policy_power:
            policy_rng = random.Random(agent_seed)
            policy = SampledBestResponsePolicy(rng=policy_rng)
            agents[power] = ObservationBestResponseAgent(power, policy=policy)
        else:
            agents[power] = RandomAgent(
                power,
                hold_probability=hold_probability,
                rng=random.Random(agent_seed),
            )

    states, titles, orders_history = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="Standard Board After Round {round}",
        stop_on_winner=stop_on_winner,
    )

    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)

    winner = states[-1].winner
    if winner is not None:
        print(f"\nWinner detected: {winner} controls a majority of supply centers.")
    elif stop_on_winner:
        print("\nNo winner within the configured round limit.")

    if visualize:
        interactive_visualize_state_mesh(states, titles)


def run_standard_board_with_deepmind_turkey(
    *,
    weights_path: str | Path,
    rounds: int = 100,
    visualize: bool = False,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    stop_on_winner: bool = True,
    temperature: float = 0.2,
) -> None:
    """Run the standard board demo with Turkey controlled by DeepMind's SL agent.

    Args:
        weights_path: Filesystem path to ``sl_params.npz`` from the public release.
        rounds: Maximum number of movement rounds to simulate.
        visualize: Whether to open the visualization mesh at the end of the run.
        seed: Optional PRNG seed shared across all agents for reproducibility.
        hold_probability: Probability a random agent issues a hold order.
        stop_on_winner: Stop early when a winner is detected.
        temperature: Softmax temperature to apply when sampling from the SL policy.
    """

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Could not find supervised-learning parameters at "
            f"{weights_path}. Download DeepMind's sl_params.npz (see diplomacy-main/README.md) "
            "and pass its full path via the weights_path argument."
        )

    state = standard_initial_state()
    base_rng = random.Random(seed)

    turkey = Power("Turkey")
    turkey_seed = base_rng.randint(0, 2**32 - 1)

    austria = Power("Austria")
    austria_seed = base_rng.randint(0, 2**32 - 1)


    # Use the DeepMind SL agent we just wrote
    turkey_agent = DeepMindSlAgent(
        power=turkey,
        sl_params_path=str(weights_path),
        rng_seed=turkey_seed,
        temperature=temperature,
    )

    austria_agent = DeepMindSlAgent(
        power=austria,
        sl_params_path=str(weights_path),
        rng_seed=austria_seed,
        temperature=temperature
    )



    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        if power == turkey:
            agents[power] = turkey_agent
            continue
        if power == austria:
            agents[power] = austria_agent
            continue
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
        title_prefix="Standard Board After Round {round}",
        stop_on_winner=stop_on_winner,
    )

    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)

    winner = states[-1].winner
    if winner is not None:
        print(f"\nWinner detected: {winner} controls a majority of supply centers.")
    elif stop_on_winner:
        print("\nNo winner within the configured round limit.")

    if visualize:
        interactive_visualize_state_mesh(states, titles)


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
    "simulate_fleet_coast_movements",
    "print_two_power_cooperation_report",
    "print_fleet_coast_demo_report",
    "print_standard_board_demo",
    "visualize_fleet_coast_demo",
    "visualize_standard_board",
    "interactive_visualize_standard_board",
    "run_standard_board_with_random_england",
    "run_standard_board_with_random_agents",
    "run_standard_board_with_deepmind_turkey",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
]


if __name__ == "__main__":
    default_weights = Path("data/sl_params.npz")
    if not default_weights.is_file():
        raise SystemExit(
            "Default weights expected at "
            f"{default_weights}. Download DeepMind's sl_params.npz (see diplomacy-main/README.md) "
            "and place it there, or call run_standard_board_with_deepmind_turkey with the correct path."
        )

    run_standard_board_with_deepmind_turkey(
        weights_path=default_weights,
        rounds=50,
        visualize=False,
        seed=123,
    )
