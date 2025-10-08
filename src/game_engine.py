from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Optional, Iterable

# New: graph + viz
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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

# ----- Game State -----
@dataclass
class GameState:
    board: Dict[str, Province]
    units: Dict[str, Unit]                 # keyed by province name
    phase: Phase = Phase.SPRING
    powers: Set[Power] = field(default_factory=set)
    graph: nx.Graph = field(default_factory=nx.Graph)

    def __post_init__(self):
        # Build/refresh the graph from the board definition
        self.graph = nx.Graph()
        for name, prov in self.board.items():
            self.graph.add_node(name, is_supply_center=prov.is_supply_center)
        for name, prov in self.board.items():
            for nbr in prov.neighbors:
                if name != nbr:
                    self.graph.add_edge(name, nbr)

    def copy(self) -> "GameState":
        s = GameState(board=self.board, units=dict(self.units), phase=self.phase, powers=set(self.powers))
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

# ----- Order helpers -----

def hold(u: Unit) -> Order:
    return Order(unit=u, type=OrderType.HOLD)

def move(u: Unit, dest: str) -> Order:
    return Order(unit=u, type=OrderType.MOVE, target=dest)

def support_hold(u: Unit, friend_loc: str) -> Order:
    return Order(unit=u, type=OrderType.SUPPORT, support_unit_loc=friend_loc)

def support_move(u: Unit, friend_from: str, friend_to: str) -> Order:
    return Order(unit=u, type=OrderType.SUPPORT, support_unit_loc=friend_from, support_target=friend_to)

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
                    moved = any(new_units.get(dest) == u and src == o.support_unit_loc and dest == o.support_target
                                 for (src, dest), u in [((ou.unit.loc, ou.target), ou.unit) for ou in orders if ou.type == OrderType.MOVE])
                    if moved:
                        resolution.succeeded.add(o)
                    else:
                        resolution.failed.add(o)

        resolution.dislodged = dislodged
        next_state = self.state.copy()
        next_state.units = new_units
        return next_state, resolution

# ----- Graph utilities & visualization -----

def build_graph(board: Dict[str, Province]) -> nx.Graph:
    G = nx.Graph()
    for name, prov in board.items():
        G.add_node(name, is_supply_center=prov.is_supply_center)
        for nbr in prov.neighbors:
            if name != nbr:
                G.add_edge(name, nbr)
    return G


def visualize_state(state: GameState, title: str = "Board State"):
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
        (x, y) = pos[loc]
        plt.text(x, y + 0.07, f"{unit.power[0]}", ha='center', va='center', fontsize=10)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----- Custom 5x3 mesh map (like your image) & demo -----

def mesh_board_5x3() -> Dict[str, Province]:
    """Build a 5x3 grid with horizontal, vertical, and diagonal (\) and (/) edges.
    Node labels chosen to match your picture where possible.
    Top row: TL, 1, 4, 2, 0
    Mid row: 6, 7, 8, 3, 9
    Bot row: BL, 13, 12, 11, 10
    TL/BL are the unlabeled corners in your image.
    Supply centers: TL, 0, 8, BL, 10 (corners + center).
    """
    names = [
        ["TL", "1", "4", "2", "0"],
        ["6", "7", "8", "3", "9"],
        ["BL", "13", "12", "11", "10"],
    ]
    # Create provinces
    board: Dict[str, Province] = {}
    for r in range(3):
        for c in range(5):
            name = names[r][c]
            is_sc = name in {"TL", "0", "8", "BL", "10"}
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
        "TL": Unit(Power("Blue"), "TL"),
        "0": Unit(Power("Pink"), "0"),
        "8": Unit(Power("Red"), "8"),
        "BL": Unit(Power("Green"), "BL"),
        "10": Unit(Power("Yellow"), "10"),
    }
    powers = {u.power for u in units.values()}
    s = GameState(board=board, units=units, powers=powers)
    return s


def _mesh_positions() -> Dict[str, Tuple[float, float]]:
    grid_pos = {
        # row 0
        "TL": (0, 2), "1": (1, 2), "4": (2, 2), "2": (3, 2), "0": (4, 2),
        # row 1
        "6": (0, 1), "7": (1, 1), "8": (2, 1), "3": (3, 1), "9": (4, 1),
        # row 2
        "BL": (0, 0), "13": (1, 0), "12": (2, 0), "11": (3, 0), "10": (4, 0),
    }
    return {k: (v[0] * 1.2, v[1] * 1.0) for k, v in grid_pos.items()}


def _draw_mesh_state(ax: plt.Axes, state: GameState, pos: Dict[str, Tuple[float, float]], title: str) -> None:
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
        ax.scatter([x], [y], s=350, c='C0', zorder=10, edgecolors='black', linewidths=1, marker='o')
        ax.text(x, y, f"{unit.power[0]}", ha='center', va='center', fontsize=10, color='white', zorder=11)
        ax.text(x, y + 0.18, f"{unit.power[0]}", ha='center', va='center', fontsize=10)

    ax.set_title(title)
    ax.axis('off')


def visualize_state_mesh(state: GameState, title: str = "5x3 Mesh Map"):
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

    pos = _mesh_positions()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    plt.subplots_adjust(bottom=0.25)

    state_index = {'value': 0}

    def draw_current() -> None:
        idx = state_index['value']
        _draw_mesh_state(ax, states[idx], pos, titles[idx])
        fig.canvas.draw_idle()

    def go_next(event) -> None:
        if state_index['value'] < len(states) - 1:
            state_index['value'] += 1
            draw_current()

    def go_prev(event) -> None:
        if state_index['value'] > 0:
            state_index['value'] -= 1
            draw_current()

    axprev = plt.axes([0.3, 0.05, 0.15, 0.08])
    axnext = plt.axes([0.55, 0.05, 0.15, 0.08])
    bprev = Button(axprev, 'Previous')
    bnext = Button(axnext, 'Next')
    bprev.on_clicked(go_prev)
    bnext.on_clicked(go_next)

    draw_current()
    plt.show()


def demo_run_mesh_with_random_orders(rounds: int = 3):
    state = demo_state_mesh()
    states = [state]
    titles = ["Initial — 5x3 Mesh Map"]

    toward = {"TL": "7", "0": "3", "BL": "13", "10": "11", "8": "8"}

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

        state, _ = Adjudicator(state).resolve(orders)
        states.append(state)
        titles.append(f"After Round {r} — 5x3 Mesh Map")

    interactive_visualize_state_mesh(states, titles)


if __name__ == "__main__":
    # still run the tiny triangle tests for sanity, then show mesh demo
    # run_self_test_and_show()
    demo_run_mesh_with_random_orders()
