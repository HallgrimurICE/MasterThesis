from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import time 
from .adjudication import Adjudicator, Resolution
from .agents import (
    Agent,
    ObservationBestResponseAgent,
    RandomAgent,
    SampledBestResponsePolicy,
)
from .maps import (
    standard_initial_state,
    triangle_initial_state,
)
from .agents.sl_agent import DeepMindSlAgent, BaselineNegotiatorAgent
from .orders import hold, move, support_hold, support_move
from .simulation import run_rounds_with_agents
from .state import GameState
from .types import Order, Power, Unit, describe_order
from .viz.mesh import interactive_visualize_state_mesh, visualize_state




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


def run_triangle_board_with_random_agents(
    rounds: int = 10,
    *,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    visualize: bool = False,
) -> None:
    """Run a short simulation on the triangle board with three random agents."""

    state = triangle_initial_state()
    rng = random.Random(seed)
    agents: Dict[Power, Agent] = {}
    for name in ("Red", "Blue", "Green"):
        power = Power(name)
        agent_seed = rng.randint(0, 2**32 - 1)
        agents[power] = RandomAgent(
            power,
            hold_probability=hold_probability,
            rng=random.Random(agent_seed),
        )

    states, titles, _ = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="Triangle Board After Round {round}",
        stop_on_winner=False,
    )
    if visualize:
        interactive_visualize_state_mesh(states, titles)

def deepmind_single_move_latency(
    *,
    weights_path: str | Path,
    k_candidates: int = 1,
    action_rollouts: int = 1,
    seed: Optional[int] = None,
) -> float:
    """Run a single DeepMindSL planning step on the standard board and return latency in seconds."""

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Could not find supervised-learning parameters at "
            f"{weights_path}. Download the SL params file and pass its full path."
        )

    # Use the same standard_initial_state as the other demos
    state = standard_initial_state()

    # Pick a deterministic power (first in sorted order)
    power = sorted(state.powers, key=str)[0]

    rng_seed = random.Random(seed).randint(0, 2**32 - 1) if seed is not None else None

    agent = DeepMindSlAgent(
        power=power,
        sl_params_path=str(weights_path),
        rng_seed=rng_seed,
        k_candidates=k_candidates,
        action_rollouts=action_rollouts,
    )

    start = time.perf_counter()
    orders = agent._plan_orders(state, round_index=0)
    duration = time.perf_counter() - start

    if not orders:
        raise RuntimeError("DeepMindSlAgent did not return any orders.")

    return duration


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

    # turkey = Power("Turkey")
    
    # turkey_seed = base_rng.randint(0, 2**32 - 1)

    austria = Power("Austria")
    austria_seed = base_rng.randint(0, 2**32 - 1)


    # Use the DeepMind SL agent we just wrote
    # turkey_agent = DeepMindSlAgent(
    #     power=turkey,
    #     sl_params_path=str(weights_path),
    #     rng_seed=turkey_seed,
    #     temperature=temperature,
    # )

    austria_agent = DeepMindSlAgent(
        power=austria,
        sl_params_path=str(weights_path),
        rng_seed=austria_seed,
        temperature=temperature
    )



    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        # if power == turkey:
        #     agents[power] = turkey_agent
        #     continue
        if power == austria:
            agents[power] = austria_agent
            continue
        agent_seed = base_rng.randint(0, 2**32 - 1)
        agents[power] = DeepMindSlAgent(
            power=power,
            sl_params_path=str(weights_path),
            rng_seed=agent_seed,
            temperature=temperature,
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


def run_standard_board_with_mixed_deepmind_and_random(
    *,
    weights_path: str | Path,
    num_games: int = 1,
    rounds: int = 10,
    visualize: bool = False,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    stop_on_winner: bool = True,
    temperature: float = 0.2,
    random_powers: Optional[List[Power]] = None,
    deepmind_configs: Optional[List[Tuple[Power, int, int]]] = None,
) -> None:
    """Run a demo with 3 RandomAgents and 4 DeepMind SL agents.

    DeepMind agents can be given distinct rollout counts (``action_rollouts``)
    and candidate widths (``k_candidates``) to compare behavior. By default the
    random agents control England, France, and Germany while the remaining
    powers use DeepMind with progressively larger search budgets. The demo can
    repeat across ``num_games`` independent matches and prints one line per
    round plus a concise summary for each run.
    """

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Could not find supervised-learning parameters at "
            f"{weights_path}. Download DeepMind's sl_params.npz (see diplomacy-main/README.md) "
            "and pass its full path via the weights_path argument."
        )

    base_rng = random.Random(seed)

    default_randoms = [Power("England"), Power("France"), Power("Germany")]
    random_powers = random_powers or default_randoms

    def build_agents(state: GameState, rng: random.Random) -> Dict[Power, Agent]:
        nonlocal deepmind_configs

        if deepmind_configs is None:
            remaining_powers = [
                p for p in sorted(state.powers, key=str) if p not in random_powers
            ]
            rollout_grid = [(2, 2), (3, 4), (4, 6), (5, 8)]
            deepmind_configs = [
                (power, k_candidates, action_rollouts)
                for power, (k_candidates, action_rollouts) in zip(
                    remaining_powers, rollout_grid
                )
            ]

        dm_by_power = {power: (k, n) for power, k, n in deepmind_configs}
        agents_for_game: Dict[Power, Agent] = {}

        for power in sorted(random_powers, key=str):
            agent_seed = rng.randint(0, 2**32 - 1)
            agents_for_game[power] = RandomAgent(
                power,
                hold_probability=hold_probability,
                rng=random.Random(agent_seed),
            )

        for power, (k_candidates, action_rollouts) in dm_by_power.items():
            agent_seed = rng.randint(0, 2**32 - 1)
            agents_for_game[power] = DeepMindSlAgent(
                power=power,
                sl_params_path=str(weights_path),
                rng_seed=agent_seed,
                temperature=temperature,
                k_candidates=k_candidates,
                action_rollouts=action_rollouts,
            )

        missing = [power for power in sorted(state.powers, key=str) if power not in agents_for_game]
        if missing:
            missing_list = ", ".join(str(p) for p in missing)
            raise ValueError(
                "Agents must be provided for every power; missing: " f"{missing_list}"
            )

        return agents_for_game

    def supply_center_summary(state: GameState) -> str:
        center_counts: Dict[Power, int] = {}
        for controller in state.supply_center_control.values():
            if controller is None:
                continue
            center_counts[controller] = center_counts.get(controller, 0) + 1
        ordered = sorted(center_counts.items(), key=lambda item: str(item[0]))
        return ", ".join(f"{power}:{count}" for power, count in ordered) or "No control"

    game_results: List[Tuple[int, Optional[Power], str]] = []

    for game_index in range(1, num_games + 1):
        state = standard_initial_state()
        agents = build_agents(state, base_rng)

        states, _, _ = run_rounds_with_agents(
            state,
            agents,
            rounds,
            title_prefix="Standard Board After Round {round}",
            stop_on_winner=stop_on_winner,
        )

        print(f"[Game {game_index}] Starting new match with {rounds} rounds.")
        for round_index, round_state in enumerate(states[1:], start=1):
            print(
                "[Game "
                f"{game_index}][Round {round_index}] "
                f"Phase={round_state.phase.name} Centers={supply_center_summary(round_state)}"
            )

        winner = states[-1].winner
        final_centers = supply_center_summary(states[-1])
        game_results.append((len(states) - 1, winner, final_centers))
        winner_display = winner if winner is not None else "No winner"
        print(
            f"[Game {game_index}] Completed after {len(states) - 1} rounds. "
            f"Winner={winner_display}. Final centers: {final_centers}."
        )

        if visualize:
            interactive_visualize_state_mesh(states, ["Round {i}" for i in range(len(states))])

    print("\n=== Mixed DeepMind/Random Summary ===")
    for idx, (rounds_played, winner, centers) in enumerate(game_results, start=1):
        winner_display = winner if winner is not None else "No winner"
        print(
            f"Game {idx}: Rounds={rounds_played}, Winner={winner_display}, Final centers={centers}"
        )

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
    "run_triangle_board_with_random_agents",
    "run_standard_board_with_mixed_deepmind_and_random",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
    "deepmind_single_move_latency"
]


if __name__ == "__main__":
    default_weights = Path("data/fppi2_params.npz")
    if not default_weights.is_file():
        raise SystemExit(
            "Default weights expected at "
            f"{default_weights}. Download DeepMind's sl_params.npz (see diplomacy-main/README.md) "
            "and place it there, or call run_standard_board_with_deepmind_turkey with the correct path."
        )


    run_standard_board_with_deepmind_turkey(
    # finish par
        weights_path=default_weights,
        rounds=100,
        visualize=False,
        seed=42,
        hold_probability=0.1,
        temperature=0.2,
    )

    # run_standard_board_with_mixed_deepmind_and_random(
    #     weights_path=default_weights,
    #     num_games=5,
    #     rounds=30,
    #     visualize=False,
    #     seed=123,
    #     hold_probability=0.1,
    #     temperature=0.2,
    # )
