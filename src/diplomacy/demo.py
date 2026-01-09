from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Set
import time 
import numpy as np
from .adjudication import Adjudicator, Resolution
from .agents import (
    Agent,
    ObservationBestResponseAgent,
    RandomAgent,
    SampledBestResponsePolicy,
    SaveBestResponseAgent,
)
from .maps import (
    standard_initial_state,
    triangle_initial_state,
)
try:
    from .agents.sl_agent import DeepMindNegotiatorAgent
except Exception:  # pragma: no cover - optional until negotiator lands
    DeepMindNegotiatorAgent = None  # type: ignore
from .agents.sl_agent import DeepMindSlAgent, DeepMindSaveAgent
from .deepmind.build_observation import build_observation
from .deepmind.actions import legal_actions_from_state, decode_actions_to_orders
from .orders import hold, move, support_hold, support_move
from .value_estimation import sl_state_value
from .simulation import run_rounds_with_agents
from .state import GameState
from .types import Order, Power, Unit, UnitType, describe_order
from .negotiation.rss import run_rss_for_power, compute_active_contracts
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

def _step_state_from_actions(state: GameState, joint_actions: Dict[Power, Iterable[int]]) -> GameState:
    orders: List[Order] = []
    for power, action_indices in joint_actions.items():
        orders.extend(
            decode_actions_to_orders(
                state=state,
                power=power,
                action_indices=action_indices,
            )
        )
    next_state, _ = Adjudicator(state).resolve(orders)
    return next_state


def _policy_fn_for_agent(agent: Agent, powers: List[Power]) -> Callable:
    policy = getattr(agent, "_policy", None)
    if policy is None:
        raise ValueError(f"Agent for {getattr(agent, 'power', 'unknown')} has no _policy for RSS.")

    def policy_fn(
        state: GameState,
        power: Power,
        legal_actions: Dict[Power, Iterable[int]],
        restricted_actions: Optional[Dict[Power, Iterable[int]]] = None,
    ) -> List[int]:
        observation = build_observation(state, last_actions=[])
        joint_legal: List[np.ndarray] = []
        for p in powers:
            allowed = legal_actions[p]
            if restricted_actions and p in restricted_actions:
                allowed = restricted_actions[p]
            joint_legal.append(np.asarray(allowed, dtype=np.int64))

        slots = list(range(len(powers)))
        actions, _ = policy.actions(
            slots_list=slots,
            observation=observation,
            legal_actions=joint_legal,
        )
        return list(actions[powers.index(power)])

    return policy_fn


def _policy_fn_from_policy(policy, powers: List[Power]) -> Callable:
    def policy_fn(
        state: GameState,
        power: Power,
        legal_actions: Dict[Power, Iterable[int]],
        restricted_actions: Optional[Dict[Power, Iterable[int]]] = None,
    ) -> List[int]:
        observation = build_observation(state, last_actions=[])
        joint_legal: List[np.ndarray] = []
        for p in powers:
            allowed = legal_actions[p]
            if restricted_actions and p in restricted_actions:
                allowed = restricted_actions[p]
            joint_legal.append(np.asarray(allowed, dtype=np.int64))

        slots = list(range(len(powers)))
        actions, _ = policy.actions(
            slots_list=slots,
            observation=observation,
            legal_actions=joint_legal,
        )
        return list(actions[powers.index(power)])

    return policy_fn


def _compute_negotiation_deals(
    state: GameState,
    agents: Dict[Power, Agent],
    negotiation_powers: List[Power],
    *,
    rss_rollouts: int = 4,
) -> Tuple[Dict[Power, Set[Power]], List]:
    powers = sorted(state.powers, key=str)
    negotiators = [p for p in powers if p in negotiation_powers]
    if not negotiators:
        return {}, []

    legal_arrays = list(legal_actions_from_state(state))
    full_power_order = sorted(state.powers, key=str)
    legal_by_power = {p: list(legal_arrays[full_power_order.index(p)]) for p in powers}

    primary_policy = next(
        (getattr(agent, "_policy", None) for agent in agents.values() if getattr(agent, "_policy", None) is not None),
        None,
    )

    policy_fns: Dict[Power, Callable] = {}
    for power in powers:
        agent = agents.get(power)
        policy = getattr(agent, "_policy", None)
        if policy is not None:
            policy_fns[power] = _policy_fn_for_agent(agent, powers)
        elif primary_policy is not None:
            policy_fns[power] = _policy_fn_from_policy(primary_policy, powers)

    # Require policy coverage for all powers so RSS can sample full joint actions.
    if len(policy_fns) != len(powers):
        print("[RSS] Skipping proposals: missing policy for some powers.")
        return {}, []

    proposals: Dict[Power, Set[Power]] = {}

    def value_fn(state_val: GameState, power_val: Power) -> float:
        policy = getattr(agents.get(power_val), "_policy", None) or primary_policy
        if policy is None:
            return 0.0
        return sl_state_value(state_val, power_val, policy=policy)

    for power in negotiators:
        proposals[power] = run_rss_for_power(
            state=state,
            power=power,
            powers=powers,
            legal_actions=legal_by_power,
            policy_fns=policy_fns,
            value_fn=value_fn,
            step_fn=_step_state_from_actions,
            rollouts=rss_rollouts,
        )

    contracts = compute_active_contracts(
        state=state,
        powers=powers,
        legal_actions=legal_by_power,
        proposals=proposals,
    )
    return proposals, contracts


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
    visualize: bool = False,
) -> None:
    """Run a short simulation on the triangle board with heuristic agents."""

    state = triangle_initial_state()
    rng = random.Random(seed)
    agents: Dict[Power, Agent] = {}
    for name in ("Red", "Blue", "Green"):
        power = Power(name)
        agent_seed = rng.randint(0, 2**32 - 1)
        policy = SampledBestResponsePolicy(rng=random.Random(agent_seed))
        agents[power] = ObservationBestResponseAgent(power, policy=policy)

    states, titles, orders_history = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="Triangle Board After Round {round}",
        stop_on_winner=False,
    )
    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)
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


def run_standard_board_with_heuristic_agents(
    rounds: int = 50,
    visualize: bool = False,
    *,
    seed: Optional[int] = None,
    stop_on_winner: bool = True,
    rollout_limit: int = 64,
    rollout_depth: int = 1,
    rollout_discount: float = 0.9,
    unit_weight: float = 1.0,
    supply_center_weight: float = 5.0,
    threatened_penalty: float = 2.0,
    base_profile_count: int = 8,
    random_ratio: float = 0.5,
    hold_probability: float = 0.2,
    heuristic_powers: Optional[List[Power]] = None,
) -> None:
    """Run the standard board with heuristic best-response and random agents."""

    state = standard_initial_state()
    base_rng = random.Random(seed)

    agents: Dict[Power, Agent] = {}
    powers = sorted(state.powers, key=str)
    if heuristic_powers is not None:
        heuristic_set = set(heuristic_powers)
        random_powers = {power for power in powers if power not in heuristic_set}
    else:
        random_count = max(0, min(len(powers), round(len(powers) * random_ratio)))
        random_powers = set(powers[:random_count])
    heuristic_powers = [power for power in powers if power not in random_powers]
    random_power_list = ", ".join(str(power) for power in sorted(random_powers, key=str))
    heuristic_power_list = ", ".join(str(power) for power in heuristic_powers)
    print("\nAgent assignments:")
    print(f"  Random agents ({len(random_powers)}): {random_power_list or '(none)'}")
    print(f"  Heuristic agents ({len(heuristic_powers)}): {heuristic_power_list or '(none)'}")
    for power in powers:
        agent_seed = base_rng.randint(0, 2**32 - 1)
        if power in random_powers:
            agents[power] = RandomAgent(
                power,
                hold_probability=hold_probability,
                rng=random.Random(agent_seed),
            )
            continue
        policy = SampledBestResponsePolicy(
            rollout_limit=rollout_limit,
            rollout_depth=rollout_depth,
            rollout_discount=rollout_discount,
            rng=random.Random(agent_seed),
            unit_weight=unit_weight,
            supply_center_weight=supply_center_weight,
            threatened_penalty=threatened_penalty,
            base_profile_count=base_profile_count,
        )
        agents[power] = ObservationBestResponseAgent(power, policy=policy)

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


def run_standard_board_heuristic_experiment(
    *,
    rounds: int = 50,
    games: int = 20,
    seed: Optional[int] = None,
    heuristic_powers: Optional[List[Power]] = None,
    hold_probability: float = 0.2,
    rollout_limit: int = 64,
    rollout_depth: int = 1,
    rollout_discount: float = 0.9,
    unit_weight: float = 1.0,
    supply_center_weight: float = 5.0,
    threatened_penalty: float = 2.0,
    base_profile_count: int = 8,
) -> None:
    """Run multiple games and average final supply centers per power."""

    state = standard_initial_state()
    powers = sorted(state.powers, key=str)
    heuristic_set = set(heuristic_powers or [])
    random_powers = {power for power in powers if power not in heuristic_set}

    random_power_list = ", ".join(str(power) for power in sorted(random_powers, key=str))
    heuristic_power_list = ", ".join(str(power) for power in sorted(heuristic_set, key=str))
    print("\nExperiment agent assignments:")
    print(f"  Random agents ({len(random_powers)}): {random_power_list or '(none)'}")
    print(f"  Heuristic agents ({len(heuristic_set)}): {heuristic_power_list or '(none)'}")

    totals: Dict[Power, int] = {power: 0 for power in powers}
    base_rng = random.Random(seed)

    for game_index in range(1, games + 1):
        game_seed = base_rng.randint(0, 2**32 - 1)
        game_rng = random.Random(game_seed)
        game_state = standard_initial_state()

        agents: Dict[Power, Agent] = {}
        for power in powers:
            agent_seed = game_rng.randint(0, 2**32 - 1)
            if power in random_powers:
                agents[power] = RandomAgent(
                    power,
                    hold_probability=hold_probability,
                    rng=random.Random(agent_seed),
                )
            else:
                policy = SampledBestResponsePolicy(
                    rollout_limit=rollout_limit,
                    rollout_depth=rollout_depth,
                    rollout_discount=rollout_discount,
                    rng=random.Random(agent_seed),
                    unit_weight=unit_weight,
                    supply_center_weight=supply_center_weight,
                    threatened_penalty=threatened_penalty,
                    base_profile_count=base_profile_count,
                )
                agents[power] = ObservationBestResponseAgent(power, policy=policy)

        states, _, _ = run_rounds_with_agents(
            game_state,
            agents,
            rounds,
            title_prefix=f"Standard Board Game {game_index} Round {{round}}",
            stop_on_winner=False,
        )

        final_state = states[-1]
        center_counts: Dict[Power, int] = {}
        for controller in final_state.supply_center_control.values():
            if controller is None:
                continue
            center_counts[controller] = center_counts.get(controller, 0) + 1

        for power in powers:
            totals[power] += center_counts.get(power, 0)

    print(f"\nAverage supply centers over {games} games:")
    for power in powers:
        average = totals[power] / float(games)
        print(f"  {power}: {average:.2f}")


def run_standard_board_with_deepmind_turkey(
    *,
    weights_path: str | Path,
    rounds: int = 100,
    visualize: bool = False,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    stop_on_winner: bool = True,
    temperature: float = 0.2,
    random_powers: Optional[List[Power]] = None,
    deepmind_configs: Optional[List[Tuple[Power, int, int]]] = None,
) -> None:
    """Run the standard board demo with three DeepMind SL agents vs four RandomAgents.

    Args:
        rounds: Maximum number of movement rounds to simulate.
        visualize: Whether to open the visualization mesh at the end of the run.
        seed: Optional PRNG seed shared across all agents for reproducibility.
        hold_probability: Probability a random agent issues a hold order.
        stop_on_winner: Stop early when a winner is detected.
        temperature: Softmax temperature to apply when sampling from the SL policy.
        random_powers: Powers to control with RandomAgent (defaults to England, France, Germany, Italy).
        deepmind_configs: Optional explicit list of (Power, k_candidates, action_rollouts) tuples;
            defaults to three DeepMind agents with configs (1,1), (1,2), (3,2) on the remaining powers.
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

    random_powers = random_powers or [
        Power("Austria"),
        Power("France"),
        Power("Russia"),
        Power("Italy"),
    ]

    if deepmind_configs is None:
        remaining_powers = [
            p for p in sorted(state.powers, key=str) if p not in random_powers
        ]
        rollout_grid = [(1, 1), (1, 2), (3, 2)]
        if len(remaining_powers) != len(rollout_grid):
            raise ValueError(
                f"Expected {len(rollout_grid)} non-random powers (got {len(remaining_powers)}). "
                f"Got: {remaining_powers}. "
                "Please provide `deepmind_configs` explicitly to adjust the agent assignments."
            )
        deepmind_configs = [
            (power, k_candidates, action_rollouts)
            for power, (k_candidates, action_rollouts) in zip(
                remaining_powers, rollout_grid
            )
        ]
    print("\nDeepMind agent configs (power, k_candidates, action_rollouts):")
    for power, k, n in deepmind_configs:
        print(f"  {power}: k_candidates={k}, action_rollouts={n}")

    dm_by_power = {power: (k, n) for power, k, n in deepmind_configs}

    agents: Dict[Power, Agent] = {}
    for power in sorted(random_powers, key=str):
        agent_seed = base_rng.randint(0, 2**32 - 1)
        agents[power] = RandomAgent(
            power,
            hold_probability=hold_probability,
            rng=random.Random(agent_seed),
        )

    for power, (k_candidates, action_rollouts) in dm_by_power.items():
        agent_seed = base_rng.randint(0, 2**32 - 1)
        agents[power] = DeepMindSaveAgent(
            power=power,
            sl_params_path=str(weights_path),
            rng_seed=agent_seed,
            temperature=temperature,
            k_candidates=k_candidates,
            action_rollouts=action_rollouts,
        )

    missing = [p for p in sorted(state.powers, key=str) if p not in agents]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise ValueError(f"Agents must be provided for every power; missing: {missing_list}")

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


def run_standard_board_br_vs_neg(
    *,
    weights_path: str | Path,
    rounds: int = 20,
    seed: Optional[int] = 42,
    hold_probability: float = 0.2,
    temperature: float = 0.1,
    k_candidates: int = 2,
    action_rollouts: int = 2,
    rss_rollouts: int = 4,
    tom_depth: int = 2,
    negotiation_powers: Optional[List[Power]] = None,
    baseline_powers: Optional[List[Power]] = None,
    stop_on_winner: bool = True,
    visualize: bool = False,
) -> None:
    """Demo with negotiation agents vs non-negotiation baselines plus randoms."""

    if DeepMindNegotiatorAgent is None:
        raise ImportError("DeepMindNegotiatorAgent not available; implement it in agents/sl_agent.py.")

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Could not find supervised-learning parameters at "
            f"{weights_path}. Download DeepMind's sl_params.npz "
            "and pass its full path via the weights_path argument."
        )

    state = standard_initial_state()
    base_rng = random.Random(seed)

    negotiation_powers = negotiation_powers 
    baseline_powers = baseline_powers 

    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        agent_seed = base_rng.randint(0, 2**32 - 1)
        if power in negotiation_powers:
            agents[power] = DeepMindNegotiatorAgent(
                power=power,
                sl_params_path=str(weights_path),
                rng_seed=agent_seed,
                temperature=temperature,
                k_candidates=k_candidates,
                action_rollouts=action_rollouts,
                rss_rollouts=rss_rollouts,
                tom_depth=tom_depth,
            )
        elif power in baseline_powers:
            agents[power] = DeepMindSaveAgent(
                power=power,
                sl_params_path=str(weights_path),
                rng_seed=agent_seed,
                temperature=temperature,
                k_candidates=k_candidates,
                action_rollouts=action_rollouts,
            )
        else:
            agents[power] = RandomAgent(
                power,
                hold_probability=hold_probability,
                rng=random.Random(agent_seed),
            )

    print("\n[br_vs_neg] Agent roster:")
    for power in sorted(state.powers, key=str):
        agent = agents.get(power)
        print(f"  {power}: {agent.__class__.__name__}")
    print(f"[br_vs_neg] Config: rounds={rounds}, k={k_candidates}, n_rollouts={action_rollouts}, rss_rollouts={rss_rollouts}")

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
    states: List[GameState] = [state]
    titles: List[str] = ["Initial State"]
    orders_history: List[List[Order]] = []

    while movement_round < rounds and (not stop_on_winner or state.winner is None):
        if state.phase.name.endswith("RETREAT"):
            print(f"[Round {movement_round}] (Retreat phase)")
            retreat_orders: List[Order] = []
            for power, agent in agents.items():
                if not any(u.power == power for u in state.pending_retreats.values()):
                    continue
                retreat_orders.extend(agent.issue_orders(state))
            state, resolution = Adjudicator(state).resolve(
                retreat_orders,
                build_callback=collect_build_choices,
            )
            states.append(state)
            titles.append(f"Round {movement_round} (Retreat)")
            orders_history.append(retreat_orders)
            if stop_on_winner and resolution.winner is not None:
                break
            continue

        movement_round += 1
        print(f"\n[Round {movement_round}] Phase={state.phase.name}")

        proposals, contracts = _compute_negotiation_deals(
            state=state,
            agents=agents,
            negotiation_powers=negotiation_powers,
            rss_rollouts=rss_rollouts,
        )
        if proposals:
            print("  Proposals:")
            for power in sorted(negotiation_powers, key=str):
                proposed = proposals.get(power, set())
                targets = ", ".join(str(p) for p in sorted(proposed, key=str)) or "(none)"
                print(f"    {power} -> {targets}")
        else:
            print("  No proposals this round (missing negotiators or policies).")

        if contracts:
            print("  Active deals:")
            for contract in contracts:
                allowed_i = sorted(int(a) for a in contract.allowed_i)
                allowed_j = sorted(int(a) for a in contract.allowed_j)
                print(
                    f"    {contract.player_i} <-> {contract.player_j} | "
                    f"{len(allowed_i)} actions for {contract.player_i}, "
                    f"{len(allowed_j)} actions for {contract.player_j}"
                )
        else:
            print("  No mutual deals this round.")

        round_orders: List[Order] = []
        for power, agent in agents.items():
            if power not in state.powers:
                continue
            agent_orders = agent.issue_orders(state)
            round_orders.extend(agent_orders)

        orders_history.append(list(round_orders))
        print("  Orders:")
        for line in _format_orders_with_actions(round_orders):
            print(line)

        state, resolution = Adjudicator(state).resolve(
            round_orders,
            build_callback=collect_build_choices,
        )
        states.append(state)
        titles.append(f"Round {movement_round}")
        print(
            f"  Resolution: winner={resolution.winner}, "
            f"auto_disbands={bool(resolution.auto_disbands)}, "
            f"auto_builds={bool(resolution.auto_builds)}"
        )
        if stop_on_winner and resolution.winner is not None:
            print(f"  Winner detected: {resolution.winner}")
            break

    final_state = states[-1]
    winner = final_state.winner
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


def run_standard_board_with_save_best_response_turkey(
    *,
    weights_path: str | Path,
    rounds: int = 10,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
    n_rollouts: int = 2,
    max_candidates: int = 8,
) -> None:
    """Run a demo with one SaveBestResponseAgent (Turkey) vs six RandomAgents."""

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Could not find supervised-learning parameters at "
            f"{weights_path}. Download DeepMind's sl_params.npz (see diplomacy-main/README.md) "
            "and pass its full path via the weights_path argument."
        )

    state = standard_initial_state()
    base_rng = random.Random(seed)

    def random_factory(power: Power) -> Agent:
        return RandomAgent(
            power,
            hold_probability=hold_probability,
            rng=random.Random(base_rng.randint(0, 2**32 - 1)),
        )

    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        if power == Power("Turkey"):
            agents[power] = SaveBestResponseAgent(
                power=power,
                opponent_factory=random_factory,
                value_kwargs={"weights_path": str(weights_path)},
                n_rollouts=n_rollouts,
                max_candidates=max_candidates,
                rng_seed=base_rng.randint(0, 2**32 - 1),
            )
        else:
            agents[power] = random_factory(power)

    states, _, _ = run_rounds_with_agents(
        state,
        agents,
        rounds,
        title_prefix="Standard Board After Round {round}",
        stop_on_winner=False,
    )

    final_state = states[-1]
    center_counts: Dict[Power, int] = {}
    for controller in final_state.supply_center_control.values():
        if controller is None:
            continue
        center_counts[controller] = center_counts.get(controller, 0) + 1

    print("\nFinal supply center counts after SaveBestResponse vs Random:")
    for power in sorted(state.powers, key=str):
        print(f"  {power}: {center_counts.get(power, 0)}")

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
    "run_standard_board_with_heuristic_agents",
    "run_standard_board_heuristic_experiment",
    "run_standard_board_with_deepmind_turkey",
    "run_triangle_board_with_random_agents",
    "run_standard_board_with_mixed_deepmind_and_random",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
    "deepmind_single_move_latency",
    "run_standard_board_with_save_best_response_turkey",
]


if __name__ == "__main__":
    # run_triangle_board_with_random_agents(rounds=10, visualize=False)
    #
    # run_standard_board_with_heuristic_agents(
    #     rounds=50,
    #     visualize=False,
    #     seed=42,
    #     rollout_depth=1,
    #     rollout_limit=32,
    #     base_profile_count=6,
    #     heuristic_powers=[Power("Russia"), Power("France"), Power("Turkey")],
    # )
    # run_standard_board_heuristic_experiment(
    #     rounds=30,
    #     games=20,
    #     seed=7,
    #     heuristic_powers=[Power("Russia"), Power("France"), Power("Turkey")],
    #     rollout_limit=24,
    #     base_profile_count=6,
    # )
    #
    # Two ToM0 negotiators vs one ToM1 negotiator (others random).
    # Requires sl_params.npz weights.
    weights_path = "data/fppi2_params.npz"
    state = standard_initial_state()
    base_rng = random.Random(42)
    agents: Dict[Power, Agent] = {}
    for power in sorted(state.powers, key=str):
        agent_seed = base_rng.randint(0, 2**32 - 1)
        if power == Power("Turkey"):
            agents[power] = DeepMindNegotiatorAgent(
                power=power,
                sl_params_path=weights_path,
                rng_seed=agent_seed,
                k_candidates=1,
                action_rollouts=1,
                rss_rollouts=1,
                tom_depth=1,
            )
        elif power in {Power("France"), Power("Russia")}:
            agents[power] = DeepMindNegotiatorAgent(
                power=power,
                sl_params_path=weights_path,
                rng_seed=agent_seed,
                k_candidates=1,
                action_rollouts=1,
                rss_rollouts=1,
                tom_depth=0,
            )
        else:
            agents[power] = RandomAgent(
                power,
                hold_probability=0.2,
                rng=random.Random(agent_seed),
            )
    run_rounds_with_agents(
        state,
        agents,
        rounds=5,
        title_prefix="Standard Board After Round {round}",
        stop_on_winner=True,
    )

    # run_standard_board_with_mixed_deepmind_and_random(
    #     weights_path=default_weights,
    #     visualize=False,
    #     rounds=50,
    #     seed=42,
    #     hold_probability=0.1,
    #     temperature=0.1,
    # )
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # run_standard_board_with_save_best_response_turkey(
    #     weights_path="data/fppi2_params.npz",
    #     rounds=20,
    #     seed=42,
    #     hold_probability=0.2,
    #     n_rollouts=2,
    #     max_candidates=2,
    # )
    # run_standard_board_br_vs_neg(
    #     weights_path="data/fppi2_params.npz",
    #     negotiation_powers=[Power("Turkey"), Power("France"), Power("Russia"), Power("Italy"), Power("England"), Power("Germany"), Power("Austria")],
    #     # baseline_powers=[],
    #     rounds=50,
    #     rss_rollouts=2,
    #     k_candidates=4,
    #     action_rollouts=2,
    #     tom_depth=2,
    # )
    # run_standard_board_br_vs_neg(
    #     weights_path="data/fppi2_params.npz",
    #     negotiation_powers=[Power("Turkey")],
    #     rounds=5,
    #     rss_rollouts=1,
    #     k_candidates=1,
    #     action_rollouts=1,
    #     tom_depth=0,
    # )
    # run_standard_board_br_vs_neg(
    #     weights_path="data/fppi2_params.npz",
    #     negotiation_powers=[Power("Turkey")],
    #     rounds=5,
    #     rss_rollouts=1,
    #     k_candidates=1,
    #     action_rollouts=1,
    #     tom_depth=1,
    # )
