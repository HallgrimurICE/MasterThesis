from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

from .graph import nx
from .state import GameState
from .types import Power, Province, Unit


def build_graph(board: Dict[str, Province]) -> nx.Graph:  # type: ignore[override]
    graph = nx.Graph()  # type: ignore[assignment]
    for name, prov in board.items():
        graph.add_node(name, is_supply_center=prov.is_supply_center)
        for nbr in prov.neighbors:
            if name != nbr:
                graph.add_edge(name, nbr)
    return graph


def square_board() -> Dict[str, Province]:
    adjacency = {
        "A": {"B", "C", "D"},
        "B": {"A", "C", "D"},
        "C": {"A", "B", "D"},
        "D": {"A", "B", "C"},
    }
    home_lookup = {
        "A": Power("Aurora"),
        "B": Power("Borealis"),
        "C": Power("Crimson"),
        "D": None,
    }
    board: Dict[str, Province] = {}
    for name, neighbors in adjacency.items():
        board[name] = Province(
            name=name,
            neighbors=set(neighbors),
            is_supply_center=True,
            home_power=home_lookup.get(name),
        )
    return board


def cooperative_attack_initial_state() -> GameState:
    board = square_board()
    units = {
        "A": Unit(Power("Aurora"), "A"),
        "B": Unit(Power("Borealis"), "B"),
        "C": Unit(Power("Crimson"), "C"),
    }
    powers = {u.power for u in units.values()}
    return GameState(board=board, units=units, powers=powers)


def mesh_board_5x3() -> Dict[str, Province]:
    names = [
        ["1", "2", "3", "4", "5"],
        ["6", "7", "8", "9", "10"],
        ["11", "12", "13", "14", "15"],
    ]
    home_lookup = {
        "1": Power("Blue"),
        "5": Power("Pink"),
        "8": Power("Red"),
        "11": Power("Green"),
        "15": Power("Yellow"),
    }
    board: Dict[str, Province] = {}
    for r in range(3):
        for c in range(5):
            name = names[r][c]
            is_sc = name in {"1", "5", "8", "11", "15"}
            board[name] = Province(
                name=name,
                neighbors=set(),
                is_supply_center=is_sc,
                home_power=home_lookup.get(name),
            )

    def in_bounds(rr: int, cc: int) -> bool:
        return 0 <= rr < 3 and 0 <= cc < 5

    for r in range(3):
        for c in range(5):
            src = names[r][c]
            nbrs: Set[str] = set()
            for dr, dc in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]:
                rr, cc = r + dr, c + dc
                if in_bounds(rr, cc):
                    nbrs.add(names[rr][cc])
            current = board[src]
            board[src] = Province(
                name=src,
                neighbors=nbrs,
                is_supply_center=current.is_supply_center,
                home_power=current.home_power,
            )
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
    return GameState(board=board, units=units, powers=powers)


__all__ = [
    "build_graph",
    "square_board",
    "cooperative_attack_initial_state",
    "mesh_board_5x3",
    "demo_state_mesh",
]
