from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Set, Tuple, Optional, Iterable, Union

import random

# New: graph + viz
try:
    import networkx as nx  # type: ignore
    NX_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback used when networkx missing
    NX_AVAILABLE = False

    class _FallbackGraph:
        """Minimal undirected graph supporting the operations used in tests."""

        def __init__(self):
            self._adjacency: Dict[str, Set[str]] = {}
            self._node_attrs: Dict[str, Dict[str, Union[bool, str]]] = {}

        def add_node(self, node: str, **attrs: Union[bool, str]) -> None:
            self._adjacency.setdefault(node, set())
            if attrs:
                existing = self._node_attrs.setdefault(node, {})
                existing.update(attrs)

        def add_edge(self, u: str, v: str) -> None:
            self.add_node(u)
            self.add_node(v)
            self._adjacency[u].add(v)
            self._adjacency[v].add(u)

        def neighbors(self, node: str) -> Iterable[str]:
            return iter(self._adjacency.get(node, set()))

        def nodes(self, data: bool = False):
            if data:
                return [
                    (node, dict(self._node_attrs.get(node, {})))
                    for node in self._adjacency
                ]
            return list(self._adjacency)

        def __getitem__(self, node: str) -> Set[str]:
            return self._adjacency.get(node, set())

    class _FallbackNX:
        Graph = _FallbackGraph

        def __getattr__(self, name: str):  # pragma: no cover - diagnostics only
            raise RuntimeError(
                "networkx is required for visualization features and advanced graph "
                f"operations (attempted to access '{name}')."
            )

    nx = _FallbackNX()  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import colors as mcolors  # type: ignore
    from matplotlib.widgets import Button  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback used when matplotlib missing
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    mcolors = None  # type: ignore
    Button = None  # type: ignore

# ============================================================
# Minimal Diplomacy-like engine (graph-based) with visualization
# Supports: HOLD, MOVE, SUPPORT-HOLD, SUPPORT-MOVE
# Rules: basic DATC-style resolution incl. support cutting, head-to-head
# ============================================================

# ----- Core Types -----
class Power(str):
    pass

@dataclass(frozen=True)
class Province:
    name: str
    # For compatibility we keep neighbors here, but the canonical topology lives in GameState.graph
    neighbors: Set[str] = field(default_factory=set)
    is_supply_center: bool = False

class Phase(Enum):
    SPRING = auto()
    FALL = auto()

class OrderType(Enum):
    HOLD = auto()
    MOVE = auto()
    SUPPORT = auto()

@dataclass(frozen=True)
class Unit:
    power: Power
    loc: str  # province name

@dataclass(frozen=True)
class Order:
    unit: Unit
    type: OrderType
    target: Optional[str] = None          # for MOVE: destination province
    support_unit_loc: Optional[str] = None # for SUPPORT: location of unit being supported (its origin)
    support_target: Optional[str] = None   # for SUPPORT-MOVE: destination being supported

    def __str__(self) -> str:
        if self.type == OrderType.HOLD:
            return f"{self.unit.power} {self.unit.loc} H"
        if self.type == OrderType.MOVE:
            return f"{self.unit.power} {self.unit.loc} -> {self.target}"
        if self.type == OrderType.SUPPORT:
            if self.support_target:
                return f"{self.unit.power} {self.unit.loc} S {self.support_unit_loc} -> {self.support_target}"
            return f"{self.unit.power} {self.unit.loc} S {self.support_unit_loc}"
        return "?"


def describe_order(order: Order) -> str:
    """Return a human-readable description of an order."""

    unit = order.unit
    if order.type == OrderType.HOLD:
        return f"{unit.power} holds position in {unit.loc}."
    if order.type == OrderType.MOVE and order.target:
        return f"{unit.power} moves from {unit.loc} to {order.target}."
    if order.type == OrderType.SUPPORT:
        if order.support_target:
            return (
                f"{unit.power} supports {order.support_unit_loc} moving to "
                f"{order.support_target}."
            )
        if order.support_unit_loc:
            return f"{unit.power} supports {order.support_unit_loc} to hold."
    return f"{unit.power} issues an order."

# ----- Game State -----
@dataclass
class GameState:
    board: Dict[str, Province]
    units: Dict[str, Unit]                 # keyed by province name
    phase: Phase = Phase.SPRING
    powers: Set[Power] = field(default_factory=set)
    graph: nx.Graph = field(default_factory=nx.Graph)
    supply_center_control: Dict[str, Optional[Power]] = field(default_factory=dict)

    def __post_init__(self):
        # Build/refresh the graph from the board definition
        self.graph = nx.Graph()
        for name, prov in self.board.items():
            self.graph.add_node(name, is_supply_center=prov.is_supply_center)
        for name, prov in self.board.items():
            for nbr in prov.neighbors:
                if name != nbr:
                    self.graph.add_edge(name, nbr)
        self._initialise_supply_center_control()

    def copy(self) -> "GameState":
        s = GameState(
            board=self.board,
            units=dict(self.units),
            phase=self.phase,
            powers=set(self.powers),
            supply_center_control=dict(self.supply_center_control),
        )
        # Graph is derived from board, so it gets rebuilt in __post_init__
        return s

    # utilities used by value/sbr later
    def supply_centers(self, p: Power) -> int:
        return sum(1 for pr, u in self.units.items() if u.power == p and self.board[pr].is_supply_center)

    def builds_available(self, p: Power) -> int:
        # Not implemented (no build phase in this minimal engine)
        return 0

    def centers_threatened(self, p: Power) -> int:
        # count your owned SCs that are neighbors to enemy units
        cnt = 0
        for pr, u in self.units.items():
            if u.power != p: continue
            if not self.board[pr].is_supply_center: continue
            if any((n in self.units and self.units[n].power != p) for n in self.graph.neighbors(pr)):
                cnt += 1
        return cnt

    # Graph-driven legal moves for a unit
    def legal_moves_from(self, province: str) -> List[str]:
        return list(self.graph.neighbors(province))

    def _initialise_supply_center_control(self) -> None:
        """Ensure every supply center has an explicit controller entry."""

        if not self.supply_center_control:
            self.supply_center_control = {}

        # Remove any stray non-supply-center entries.
        for name in list(self.supply_center_control.keys()):
            province = self.board.get(name)
            if province is None or not province.is_supply_center:
                self.supply_center_control.pop(name, None)

        for name, prov in self.board.items():
            if prov.is_supply_center:
                self.supply_center_control.setdefault(name, None)

    def update_supply_center_control(self, prev_phase: Phase) -> None:
        """Update controller assignments if the previous phase was Fall."""

        if prev_phase != Phase.FALL:
            return

        for loc, prov in self.board.items():
            if not prov.is_supply_center:
                continue
            occupying_unit = self.units.get(loc)
            if occupying_unit is not None:
                self.supply_center_control[loc] = occupying_unit.power

# ----- Order helpers -----

def hold(u: Unit) -> Order:
    return Order(unit=u, type=OrderType.HOLD)

def move(u: Unit, dest: str) -> Order:
    return Order(unit=u, type=OrderType.MOVE, target=dest)

def support_hold(u: Unit, friend_loc: str) -> Order:
    return Order(unit=u, type=OrderType.SUPPORT, support_unit_loc=friend_loc)

def support_move(u: Unit, friend_from: str, friend_to: str) -> Order:
    return Order(unit=u, type=OrderType.SUPPORT, support_unit_loc=friend_from, support_target=friend_to)

# ----- Agent helpers -----


class Agent:
    """Base class for programmable agents that can issue orders each round."""

    def __init__(self, power: Power):
        self.power = power
        self._round_index = 0

    def issue_orders(self, state: "GameState") -> List[Order]:
        """Return this power's orders for the current round.

        Subclasses should implement :meth:`_plan_orders` to describe their behaviour.
        """

        planned = self._plan_orders(state, self._round_index)
        self._round_index += 1

        for order in planned:
            if order.unit.power != self.power:
                raise ValueError(
                    f"Agent for power {self.power} produced an order for {order.unit.power}."
                )
        # Ensure at most one order per unit; duplicates would break adjudication assumptions.
        seen_units: Set[str] = set()
        for order in planned:
            if order.unit.loc in seen_units:
                raise ValueError(
                    f"Multiple orders issued for unit currently in {order.unit.loc}."
                )
            seen_units.add(order.unit.loc)
        return planned

    def _plan_orders(self, state: "GameState", round_index: int) -> List[Order]:
        raise NotImplementedError


Directive = Union[str, Order, None, Callable[[Unit, "GameState"], Order]]


class ScriptedAgent(Agent):
    """Agent whose behaviour is programmed via a per-round script.

    The script maps the (zero-indexed) round number to orders for units currently
    controlled by the power. Orders can be provided as:

    - ``"H"`` / ``"HOLD"`` / ``None`` – keep the unit in place.
    - Province names (``str``) – attempt to move to that province if legal.
    - Callables ``(unit, state) -> Order`` – useful for more complex directives.
    - Explicit :class:`Order` objects – full control, e.g., for support orders.

    Unscripted units or invalid destinations default to HOLD.
    """

    def __init__(self, power: Power, script: Dict[int, Dict[str, Directive]]):
        super().__init__(power)
        self.script = script

    def _plan_orders(self, state: "GameState", round_index: int) -> List[Order]:
        orders: List[Order] = []
        planned_orders = self.script.get(round_index, {})
        for unit in state.units.values():
            if unit.power != self.power:
                continue

            directive = planned_orders.get(unit.loc)
            if callable(directive):
                orders.append(directive(unit, state))
                continue
            if isinstance(directive, Order):
                orders.append(directive)
                continue

            if directive is None:
                orders.append(hold(unit))
                continue

            if isinstance(directive, str):
                if directive.upper() in {"H", "HOLD"}:
                    orders.append(hold(unit))
                    continue
                if directive in state.legal_moves_from(unit.loc):
                    orders.append(move(unit, directive))
                    continue

            # Fallback: illegal or unrecognised directive
            orders.append(hold(unit))

        return orders


class RandomAgent(Agent):
    """Agent that issues random legal orders without any scripted behaviour."""

    def __init__(
        self,
        power: Power,
        *,
        hold_probability: float = 0.2,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(power)
        if not 0.0 <= hold_probability <= 1.0:
            raise ValueError("hold_probability must be between 0 and 1 inclusive")
        self.hold_probability = hold_probability
        self._rng = rng or random.Random()

    def _plan_orders(self, state: "GameState", round_index: int) -> List[Order]:
        orders: List[Order] = []
        for unit in state.units.values():
            if unit.power != self.power:
                continue

            legal_moves = state.legal_moves_from(unit.loc)
            choose_hold = (
                not legal_moves
                or self._rng.random() < self.hold_probability
            )

            if choose_hold:
                orders.append(hold(unit))
            else:
                destination = self._rng.choice(legal_moves)
                orders.append(move(unit, destination))

        return orders

# ----- Adjudicator -----
@dataclass
class Resolution:
    succeeded: Set[Order] = field(default_factory=set)
    failed: Set[Order] = field(default_factory=set)
    dislodged: Set[str] = field(default_factory=set)  # provinces where unit was dislodged

class Adjudicator:
    def __init__(self, state: GameState):
        self.state = state

    def resolve(self, orders: List[Order]) -> Tuple[GameState, Resolution]:
        """Resolve simultaneous orders without convoys.
        Implements: movement resolution, support counting, support cutting, head-to-head, self-dislodgement prohibition.
        """
        # Index orders
        by_loc: Dict[str, Order] = {o.unit.loc: o for o in orders}
        # Any units without an order HOLD by default
        for loc, unit in self.state.units.items():
            if loc not in by_loc:
                by_loc[loc] = hold(unit)
        orders = list(by_loc.values())

        # 1) Build attack map (who attacks where) and hold/support sets
        attacks_to: Dict[str, List[Order]] = {}
        supports: List[Order] = []
        for o in orders:
            if o.type == OrderType.MOVE:
                # Only allow graph-legal moves
                if o.target not in self.state.legal_moves_from(o.unit.loc):
                    continue
                attacks_to.setdefault(o.target, []).append(o)
            elif o.type == OrderType.SUPPORT:
                supports.append(o)

        # 2) Determine valid supports (not cut)
        valid_supports: Set[Order] = set()
        for s in supports:
            sup_prov = s.unit.loc
            attackers = attacks_to.get(sup_prov, [])
            cut = False
            for atk in attackers:
                if s.support_target is not None:
                    # supporting a move friend_from -> friend_to
                    if not (atk.unit.loc == s.support_unit_loc and atk.target == s.support_target):
                        cut = True
                        break
                else:
                    # supporting a hold at support_unit_loc
                    if not (atk.unit.loc == s.support_unit_loc and atk.type == OrderType.MOVE and atk.target == sup_prov):
                        cut = True
                        break
            if not cut:
                valid_supports.add(s)

        # 3) Compute strengths of all attacks and holds
        hold_strength: Dict[str, int] = {loc: 1 for loc in self.state.units.keys()}
        attack_strength: Dict[Tuple[str, str], int] = {}

        # add supports to holds
        for s in valid_supports:
            if s.support_target is None and s.support_unit_loc in self.state.units:
                hold_strength[s.support_unit_loc] = hold_strength.get(s.support_unit_loc, 1) + 1

        # attack strengths
        for o in orders:
            if o.type == OrderType.MOVE and o.target in self.state.graph[o.unit.loc]:
                attack_strength[(o.unit.loc, o.target)] = 1
        for s in valid_supports:
            if s.support_target is not None:
                key = (s.support_unit_loc, s.support_target)
                if key in attack_strength:
                    attack_strength[key] += 1

        # 4) Resolve moves by destination (bounces, dislodgements, head-to-head)
        resolution = Resolution()
        new_units: Dict[str, Unit] = dict(self.state.units)
        dislodged: Set[str] = set()

        winners_by_dest: Dict[str, List[Tuple[Order, int]]] = {}
        for dest, incoming in attacks_to.items():
            candidates = []
            for o in incoming:
                key = (o.unit.loc, o.target)
                if key not in attack_strength:
                    continue
                candidates.append((o, attack_strength[key]))
            if candidates:
                max_s = max(s for _, s in candidates)
                winners_by_dest[dest] = [(o, s) for (o, s) in candidates if s == max_s]

        for dest, winners in winners_by_dest.items():
            dest_unit = self.state.units.get(dest)
            if len(winners) > 1:
                # bounce
                continue
            winner, w_s = winners[0]
            # head-to-head?
            head_to_head = False
            if dest_unit is not None:
                opp_order = by_loc.get(dest)
                if opp_order and opp_order.type == OrderType.MOVE and opp_order.target == winner.unit.loc:
                    head_to_head = True
            if head_to_head:
                atk_ab = w_s
                atk_ba = attack_strength.get((dest, winner.unit.loc), 0)
                if atk_ab > atk_ba:
                    dislodged.add(dest)
                    new_units.pop(winner.unit.loc, None)
                    new_units[dest] = winner.unit
                elif atk_ba > atk_ab:
                    # opponent wins; winner fails to move
                    continue
                else:
                    # equal: bounce
                    continue
            else:
                defense = hold_strength.get(dest, 0)
                if dest_unit is not None and dest_unit.power == winner.unit.power:
                    continue  # self-dislodgement prohibited
                if w_s > defense:
                    if dest_unit is not None:
                        dislodged.add(dest)
                    new_units.pop(winner.unit.loc, None)
                    new_units[dest] = winner.unit
                else:
                    pass

        # mark success/failure sets
        for o in orders:
            if o.type == OrderType.MOVE:
                if new_units.get(o.target) == o.unit:
                    resolution.succeeded.add(o)
                else:
                    resolution.failed.add(o)
            elif o.type == OrderType.HOLD:
                if o.unit.loc not in dislodged:
                    resolution.succeeded.add(o)
                else:
                    resolution.failed.add(o)
            else:  # SUPPORT
                if o.support_target is None:
                    if o.support_unit_loc not in dislodged:
                        resolution.succeeded.add(o)
                    else:
                        resolution.failed.add(o)
                else:
                    moved = any(
                        new_units.get(dest) == u
                        and src == o.support_unit_loc
                        and dest == o.support_target
                        for (src, dest), u in [
                            ((ou.unit.loc, ou.target), ou.unit)
                            for ou in orders
                            if ou.type == OrderType.MOVE
                        ]
                    )
                    if moved:
                        resolution.succeeded.add(o)
                    else:
                        resolution.failed.add(o)

        resolution.dislodged = dislodged
        normalized_units: Dict[str, Unit] = {
            loc: Unit(unit.power, loc)
            for loc, unit in new_units.items()
        }
        next_state = self.state.copy()
        next_state.units = normalized_units
        prev_phase = self.state.phase
        next_state.phase = Phase.FALL if prev_phase == Phase.SPRING else Phase.SPRING
        next_state.update_supply_center_control(prev_phase=prev_phase)
        return next_state, resolution

# ----- Game runners -----


def run_rounds_with_agents(
    initial_state: GameState,
    agents: Dict[Power, Agent],
    rounds: int,
    *,
    title_prefix: str = "After Round {round}",
) -> Tuple[List[GameState], List[str], List[List[Order]]]:
    """Execute a number of rounds using programmable agents.

    Parameters
    ----------
    initial_state:
        Starting :class:`GameState`.
    agents:
        Mapping from power to the :class:`Agent` responsible for issuing its orders.
        Powers without a registered agent simply HOLD each round.
    rounds:
        Number of rounds to simulate.
    title_prefix:
        Format string used when generating titles for visualisation; the placeholder
        ``{round}`` is replaced with the one-indexed round number.

    Returns
    -------
    (states, titles, orders_by_round):
        A tuple containing the sequence of game states (including the initial state),
        corresponding titles, and the list of orders issued in each round.
    """

    state = initial_state
    states = [state]
    titles = ["Initial State"]
    orders_history: List[List[Order]] = []

    for round_idx in range(1, rounds + 1):
        round_orders: List[Order] = []
        for power, agent in agents.items():
            if power not in state.powers:
                continue
            agent_orders = agent.issue_orders(state)
            round_orders.extend(agent_orders)

        orders_history.append(list(round_orders))
        state, _ = Adjudicator(state).resolve(round_orders)
        states.append(state)
        titles.append(title_prefix.format(round=round_idx))

    return states, titles, orders_history

# ----- Graph utilities & visualization -----

def build_graph(board: Dict[str, Province]) -> nx.Graph:
    G = nx.Graph()
    for name, prov in board.items():
        G.add_node(name, is_supply_center=prov.is_supply_center)
        for nbr in prov.neighbors:
            if name != nbr:
                G.add_edge(name, nbr)
    return G


POWER_COLORS: Dict[str, str] = {
    "Blue": "#1f77b4",
    "Pink": "#ff69b4",
    "Red": "#d62728",
    "Green": "#2ca02c",
    "Yellow": "#ffbf00",
}


def _power_color(power: Power) -> str:
    return POWER_COLORS.get(str(power), "#666666")


def _power_text_color(color: str) -> str:
    if not MATPLOTLIB_AVAILABLE or mcolors is None:
        return "white"
    try:
        rgb = mcolors.to_rgb(color)
    except ValueError:
        return "white"
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.6 else "white"


def visualize_state(state: GameState, title: str = "Board State"):
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError(
            "networkx and matplotlib are required for visualization utilities."
        )
    # Layout: fixed for stability (spring layout is fine for small maps)
    pos = nx.spring_layout(state.graph, seed=7)

    # Node colors: SC vs non-SC
    sc_nodes = [n for n, d in state.graph.nodes(data=True) if d.get('is_supply_center', False)]
    non_sc = [n for n in state.graph.nodes() if n not in sc_nodes]

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_edges(state.graph, pos)
    nx.draw_networkx_nodes(state.graph, pos, nodelist=non_sc, node_color='white')
    nx.draw_networkx_nodes(state.graph, pos, nodelist=sc_nodes, node_shape='s')
    nx.draw_networkx_labels(state.graph, pos, font_size=10)

    # Draw units as annotations on nodes
    for loc, unit in state.units.items():
        x, y = pos[loc]
        color = _power_color(unit.power)
        text_color = _power_text_color(color)
        plt.scatter([x], [y], s=350, c=color, zorder=10, edgecolors='black', linewidths=1, marker='o')
        plt.text(x, y, f"{unit.power[0]}", ha='center', va='center', fontsize=10, color=text_color, zorder=11)
        plt.text(x, y + 0.07, f"{unit.power}", ha='center', va='center', fontsize=9)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----- Cooperative attack scenario (triangle map) -----


def _triangle_board() -> Dict[str, Province]:
    """Create a fully connected three-province map for cooperation demos."""

    adjacency = {
        "A": {"B", "C"},
        "B": {"A", "C"},
        "C": {"A", "B"},
    }
    board: Dict[str, Province] = {}
    for name, neighbors in adjacency.items():
        board[name] = Province(
            name=name,
            neighbors=set(neighbors),
            is_supply_center=True,
        )
    return board


def cooperative_attack_initial_state() -> GameState:
    """Initialise three powers poised around a shared target province."""

    board = _triangle_board()
    units = {
        "A": Unit(Power("Aurora"), "A"),
        "B": Unit(Power("Borealis"), "B"),
        "C": Unit(Power("Crimson"), "C"),
    }
    powers = {u.power for u in units.values()}
    return GameState(board=board, units=units, powers=powers)


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
    """Render orders alongside their human-readable descriptions."""

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
    """Emit a textual description of the cooperative attack scenario."""

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
        occupying = {
            loc: unit.power for loc, unit in sorted(next_state.units.items())
        }
        print("Post-resolution occupants:")
        for loc, power in occupying.items():
            print(f"  - {loc}: {power}")


# ----- Custom 5x3 mesh map (like your image) & demo -----

def mesh_board_5x3() -> Dict[str, Province]:
    """Build a 5x3 grid with horizontal, vertical, and diagonal (\\) and (/) edges.
    Node labels chosen to match your picture where possible.
    Top row: TL, 1, 4, 2, 0
    Mid row: 6, 7, 8, 3, 9
    Bot row: BL, 13, 12, 11, 10
    TL/BL are the unlabeled corners in your image.
    """
    names = [
        ["1", "2", "3", "4", "5"],
        ["6", "7", "8", "9", "10"],
        ["11", "12", "13", "14", "15"],
    ]
    # Create provinces
    board: Dict[str, Province] = {}
    for r in range(3):
        for c in range(5):
            name = names[r][c]
            is_sc = name in {"1", "5", "8", "11", "15"}
            board[name] = Province(name=name, is_supply_center=is_sc)
    # Add neighbors (grid + diagonals)
    def in_bounds(rr, cc):
        return 0 <= rr < 3 and 0 <= cc < 5
    for r in range(3):
        for c in range(5):
            src = names[r][c]
            nbrs: Set[str] = set()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0),  # 4-neighborhood
                           (1,1), (-1,-1), (1,-1), (-1,1)]:  # diagonals
                rr, cc = r+dr, c+dc
                if in_bounds(rr, cc):
                    nbrs.add(names[rr][cc])
            board[src] = Province(name=src, neighbors=nbrs, is_supply_center=board[src].is_supply_center)
    return board


def demo_state_mesh() -> GameState:
    board = mesh_board_5x3()
    units = {
        "1": Unit(Power("Blue"), "1"),
        "5": Unit(Power("Pink"), "5"),
        "8": Unit(Power("Red"), "8"),
        "11": Unit(Power("Green"), "11"),
        "15": Unit(Power("Yellow"), "15"),
    }
    powers = {u.power for u in units.values()}
    s = GameState(board=board, units=units, powers=powers)
    return s


def _mesh_positions() -> Dict[str, Tuple[float, float]]:
    grid_pos = {
        # row 0
        "1": (0, 2), "2": (1, 2), "3": (2, 2), "4": (3, 2), "5": (4, 2),
        # row 1
        "6": (0, 1), "7": (1, 1), "8": (2, 1), "9": (3, 1), "10": (4, 1),
        # row 2
        "11": (0, 0), "12": (1, 0), "13": (2, 0), "14": (3, 0), "15": (4, 0),
    }
    return {k: (v[0] * 1.2, v[1] * 1.0) for k, v in grid_pos.items()}


def _draw_mesh_state(ax: plt.Axes, state: GameState, pos: Dict[str, Tuple[float, float]], title: str) -> None:
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError(
            "networkx and matplotlib are required for visualization utilities."
        )
    ax.clear()
    nx.draw_networkx_edges(state.graph, pos, ax=ax)

    sc_nodes = [n for n, d in state.graph.nodes(data=True) if d.get('is_supply_center', False)]
    non_sc = [n for n in state.graph.nodes() if n not in sc_nodes]
    nx.draw_networkx_nodes(
        state.graph,
        pos,
        nodelist=non_sc,
        node_color='white',
        edgecolors='black',
        linewidths=1,
        node_size=1200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        state.graph,
        pos,
        nodelist=sc_nodes,
        node_color='white',
        edgecolors='red',
        linewidths=1,
        node_size=1200,
        ax=ax,
    )
    nx.draw_networkx_labels(state.graph, pos, font_size=9, ax=ax)

    for loc, unit in state.units.items():
        x, y = pos[loc]
        color = _power_color(unit.power)
        text_color = _power_text_color(color)
        ax.scatter([x], [y], s=350, c=color, zorder=10, edgecolors='black', linewidths=1, marker='o')
        ax.text(x, y, f"{unit.power[0]}", ha='center', va='center', fontsize=10, color=text_color, zorder=11)
        ax.text(x, y + 0.18, f"{unit.power}", ha='center', va='center', fontsize=9)

    ax.set_title(title)
    ax.axis('off')


def visualize_state_mesh(state: GameState, title: str = "5x3 Mesh Map"):
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError(
            "networkx and matplotlib are required for visualization utilities."
        )
    pos = _mesh_positions()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _draw_mesh_state(ax, state, pos, title)
    plt.tight_layout()
    plt.show()


def interactive_visualize_state_mesh(states: List[GameState], titles: Optional[List[str]] = None) -> None:
    if not states:
        return

    if titles is None or len(titles) != len(states):
        titles = [f"Round {i}" for i in range(len(states))]

    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None or Button is None:
        raise RuntimeError(
            "networkx and matplotlib are required for visualization utilities."
        )

    pos = _mesh_positions()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    plt.subplots_adjust(bottom=0.25)

    state_index = {'value': 0}

    def _set_button_enabled(button: Button, *, enabled: bool, base_color: str, base_hover: str) -> None:
        button.eventson = enabled
        face_color = base_color if enabled else '#d3d3d3'
        hover_color = base_hover if enabled else '#c0c0c0'
        button.ax.patch.set_facecolor(face_color)
        button.hovercolor = hover_color
        label_color = 'black' if enabled else '#6f6f6f'
        button.label.set_color(label_color)

    button_styles: Dict[str, Tuple[str, str]] = {}

    axprev = plt.axes([0.3, 0.05, 0.15, 0.08])
    axnext = plt.axes([0.55, 0.05, 0.15, 0.08])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')

    button_styles['prev'] = (bprev.color, bprev.hovercolor)
    button_styles['next'] = (bnext.color, bnext.hovercolor)

    def update_buttons() -> None:
        at_start = state_index['value'] == 0
        at_end = state_index['value'] == len(states) - 1
        base_prev, base_prev_hover = button_styles['prev']
        base_next, base_next_hover = button_styles['next']
        _set_button_enabled(bprev, enabled=not at_start, base_color=base_prev, base_hover=base_prev_hover)
        _set_button_enabled(bnext, enabled=not at_end, base_color=base_next, base_hover=base_next_hover)

    def draw_current() -> None:
        idx = state_index['value']
        _draw_mesh_state(ax, states[idx], pos, titles[idx])
        update_buttons()
        fig.canvas.draw_idle()

    def go_next(event) -> None:
        if state_index['value'] < len(states) - 1:
            state_index['value'] += 1
            draw_current()

    def go_prev(event) -> None:
        if state_index['value'] > 0:
            state_index['value'] -= 1
            draw_current()

    bprev.on_clicked(go_prev)
    bnext.on_clicked(go_next)

    draw_current()
    plt.show()


def demo_run_mesh_with_random_orders(rounds: int = 3):
    state = demo_state_mesh()
    states = [state]
    titles = ["Initial — 5x3 Mesh Map"]

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
        titles.append(f"After Round {r} — 5x3 Mesh Map")

    interactive_visualize_state_mesh(states, titles)


def demo_run_mesh_with_random_agents(
    rounds: int = 5,
    *,
    seed: Optional[int] = None,
    hold_probability: float = 0.2,
) -> None:
    """Run the 5x3 mesh map using autonomous random agents with visualisation."""

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
        title_prefix="After Round {round} — Random Agents on 5x3 Mesh",
    )

    for round_index, orders in enumerate(orders_history, start=1):
        print(f"\nRound {round_index} orders:")
        for line in _format_orders_with_actions(orders):
            print(line)

    interactive_visualize_state_mesh(states, titles)


# def demo_run_mesh_with_scripted_agents(rounds: int = 3) -> None:
#     """Showcase programmable agents on the 5x3 mesh map."""

#     state = demo_state_mesh()

#     blue_agent = ScriptedAgent(
#         Power("Blue"),
#         {
#             0: {"TL": "7"},
#             1: {"7": "8"},
#             2: {"8": "8"},
#         },
#     )
#     pink_agent = ScriptedAgent(
#         Power("Pink"),
#         {
#             0: {"0": "3"},
#             1: {"3": "8"},
#         },
#     )
#     green_agent = ScriptedAgent(
#         Power("Green"),
#         {
#             0: {"BL": "13"},
#             1: {"13": "8"},
#         },
#     )
#     yellow_agent = ScriptedAgent(
#         Power("Yellow"),
#         {
#             0: {"10": "11"},
#             1: {"11": "8"},
#         },
#     )

#     agents: Dict[Power, Agent] = {
#         blue_agent.power: blue_agent,
#         pink_agent.power: pink_agent,
#         green_agent.power: green_agent,
#         yellow_agent.power: yellow_agent,
#     }

#     states, titles, _ = run_rounds_with_agents(
#         state,
#         agents,
#         rounds,
#         title_prefix="After Round {round} — Scripted 5x3 Mesh",
#     )

#     interactive_visualize_state_mesh(states, titles)


if __name__ == "__main__":
    # still run the tiny triangle tests for sanity, then show mesh demo
    # run_self_test_and_show()
    print_two_power_cooperation_report()
    demo_run_mesh_with_random_agents()
