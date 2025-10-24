from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .graph import nx
from .types import OrderType, Phase, Power, Province, Unit


@dataclass
class GameState:
    board: Dict[str, Province]
    units: Dict[str, Unit]  # keyed by province name
    phase: Phase = Phase.SPRING
    powers: Set[Power] = field(default_factory=set)
    graph: nx.Graph = field(default_factory=nx.Graph)  # type: ignore[assignment]
    supply_center_control: Dict[str, Optional[Power]] = field(default_factory=dict)
    pending_retreats: Dict[str, Unit] = field(default_factory=dict)
    retreat_forbidden: Dict[str, Set[str]] = field(default_factory=dict)
    contested_provinces: Set[str] = field(default_factory=set)
    supply_update_due: bool = False

    def __post_init__(self) -> None:
        # Build/refresh the graph from the board definition
        self.graph = nx.Graph()  # type: ignore[assignment]
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
            pending_retreats=dict(self.pending_retreats),
            retreat_forbidden={k: set(v) for k, v in self.retreat_forbidden.items()},
            contested_provinces=set(self.contested_provinces),
            supply_update_due=self.supply_update_due,
        )
        return s

    # utilities used by value/sbr later
    def supply_centers(self, power: Power) -> int:
        return sum(
            1
            for province, unit in self.units.items()
            if unit.power == power and self.board[province].is_supply_center
        )

    def builds_available(self, power: Power) -> int:
        # Not implemented (no build phase in this minimal engine)
        return 0

    def centers_threatened(self, power: Power) -> int:
        # count your owned SCs that are neighbors to enemy units
        cnt = 0
        for province, unit in self.units.items():
            if unit.power != power:
                continue
            if not self.board[province].is_supply_center:
                continue
            if any(
                (neighbor in self.units and self.units[neighbor].power != power)
                for neighbor in self.graph.neighbors(province)
            ):
                cnt += 1
        return cnt

    def legal_moves_from(self, province: str) -> List[str]:
        return list(self.graph.neighbors(province))

    def legal_retreats_from(self, province: str) -> List[str]:
        """Return admissible retreat destinations for the dislodged unit at ``province``."""

        if province not in self.pending_retreats:
            return []

        occupied = set(self.units.keys())
        forbidden = set(self.retreat_forbidden.get(province, set()))
        contested = self.contested_provinces

        legal: List[str] = []
        for neighbor in self.graph.neighbors(province):
            if neighbor in occupied:
                continue
            if neighbor in forbidden:
                continue
            if neighbor in contested:
                continue
            legal.append(neighbor)
        return legal

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


__all__ = ["GameState"]
