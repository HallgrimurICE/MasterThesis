from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .graph import nx
from .types import (
    OrderType,
    Phase,
    Power,
    Province,
    ProvinceType,
    Unit,
    UnitType,
)


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
    winner: Optional[Power] = None
    pending_disbands: Dict[Power, int] = field(default_factory=dict)
    pending_builds: Dict[Power, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Build/refresh the graph from the board definition
        self.graph = nx.Graph()  # type: ignore[assignment]
        for name, prov in self.board.items():
            self.graph.add_node(
                name,
                is_supply_center=prov.is_supply_center,
                province_type=prov.province_type,
            )
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
            winner=self.winner,
            pending_disbands=dict(self.pending_disbands),
            pending_builds=dict(self.pending_builds),
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
        unit = self.units.get(province)
        if unit is None:
            return []
        legal: List[str] = []
        for neighbor in self.graph.neighbors(province):
            destination = self.board.get(neighbor)
            if destination is None:
                continue
            if self._unit_can_enter(unit.unit_type, destination.province_type):
                legal.append(neighbor)
        return legal

    def legal_retreats_from(self, province: str) -> List[str]:
        """Return admissible retreat destinations for the dislodged unit at ``province``."""

        if province not in self.pending_retreats:
            return []

        occupied = set(self.units.keys())
        forbidden = set(self.retreat_forbidden.get(province, set()))
        contested = self.contested_provinces

        unit = self.pending_retreats[province]
        legal: List[str] = []
        for neighbor in self.graph.neighbors(province):
            if neighbor in occupied:
                continue
            if neighbor in forbidden:
                continue
            if neighbor in contested:
                continue
            destination = self.board.get(neighbor)
            if destination is None:
                continue
            if not self._unit_can_enter(unit.unit_type, destination.province_type):
                continue
            legal.append(neighbor)
        return legal

    @staticmethod
    def _unit_can_enter(unit_type: UnitType, province_type: ProvinceType) -> bool:
        if unit_type == UnitType.ARMY:
            return province_type in {ProvinceType.LAND, ProvinceType.COAST}
        if unit_type == UnitType.FLEET:
            return province_type in {ProvinceType.COAST, ProvinceType.SEA}
        return False

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
                if name not in self.supply_center_control:
                    occupant = self.units.get(name)
                    self.supply_center_control[name] = occupant.power if occupant else None

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

    def update_pending_disbands(self) -> None:
        """Compute how many units each power must remove due to supply shortages."""

        counts: Dict[Power, int] = {}
        for controller in self.supply_center_control.values():
            if controller is None:
                continue
            counts[controller] = counts.get(controller, 0) + 1

        unit_counts: Dict[Power, int] = {}
        for unit in self.units.values():
            unit_counts[unit.power] = unit_counts.get(unit.power, 0) + 1

        disbands: Dict[Power, int] = {}
        for power, units_owned in unit_counts.items():
            allowed = counts.get(power, 0)
            deficit = units_owned - allowed
            if deficit > 0:
                disbands[power] = deficit

        self.pending_disbands = disbands
        if disbands:
            self.pending_builds.clear()

    def auto_disband(self) -> Dict[Power, List[str]]:
        """Automatically remove units to satisfy pending disband requirements.

        Units not on supply centers are removed first (alphabetically by province),
        followed by those on supply centers if still needed.
        """

        removed: Dict[Power, List[str]] = {}
        for power, count in list(self.pending_disbands.items()):
            if count <= 0:
                continue
            provinces = [loc for loc, unit in self.units.items() if unit.power == power]
            if not provinces:
                continue
            non_sc = sorted(
                [loc for loc in provinces if not self.board[loc].is_supply_center]
            )
            sc = sorted(
                [loc for loc in provinces if self.board[loc].is_supply_center]
            )
            removal_order = non_sc + sc
            to_remove = removal_order[:count]
            if not to_remove:
                continue
            for loc in to_remove:
                self.units.pop(loc, None)
            removed[power] = to_remove
        if removed:
            self.pending_disbands = {}
        return removed

    def update_pending_builds(self) -> None:
        """Determine build opportunities after disbands have been resolved."""

        if self.pending_disbands:
            self.pending_builds = {}
            return

        supply_counts: Dict[Power, int] = {}
        for controller in self.supply_center_control.values():
            if controller is None:
                continue
            supply_counts[controller] = supply_counts.get(controller, 0) + 1

        unit_counts: Dict[Power, int] = {}
        for unit in self.units.values():
            unit_counts[unit.power] = unit_counts.get(unit.power, 0) + 1

        builds: Dict[Power, int] = {}
        for power, supply_total in supply_counts.items():
            diff = supply_total - unit_counts.get(power, 0)
            if diff > 0:
                builds[power] = diff

        self.pending_builds = builds

    def available_build_sites(self, power: Power) -> List[str]:
        return [
            loc
            for loc, controller in self.supply_center_control.items()
            if controller == power
            and self.board[loc].home_power == power
            and loc not in self.units
        ]

    def apply_build_choices(
        self,
        selections: Mapping[Power, Sequence[Tuple[str, UnitType]]],
    ) -> Dict[Power, List[str]]:
        built: Dict[Power, List[str]] = {}
        for power, builds in selections.items():
            allowed = self.pending_builds.get(power, 0)
            if allowed <= 0:
                continue
            available = set(self.available_build_sites(power))
            if not available:
                continue
            placed: List[str] = []
            for loc, unit_type in builds:
                if len(placed) >= allowed:
                    break
                if loc not in available:
                    continue
                province = self.board.get(loc)
                if province is None:
                    continue
                if unit_type == UnitType.FLEET and province.province_type == ProvinceType.LAND:
                    continue
                if unit_type == UnitType.ARMY and province.province_type == ProvinceType.SEA:
                    continue
                self.units[loc] = Unit(power, loc, unit_type)
                available.remove(loc)
                placed.append(loc)
            if not placed:
                continue
            built[power] = placed
            remaining = allowed - len(placed)
            if remaining > 0:
                self.pending_builds[power] = remaining
            else:
                self.pending_builds.pop(power, None)
        return built

    def auto_build(self) -> Dict[Power, List[str]]:
        """Automatically build units at eligible home supply centers."""

        built: Dict[Power, List[str]] = {}
        for power, count in list(self.pending_builds.items()):
            if count <= 0:
                continue
            candidates = self.available_build_sites(power)
            if not candidates:
                continue
            candidates.sort()
            selection = candidates[:count]
            if not selection:
                continue
            for loc in selection:
                province = self.board.get(loc)
                unit_type = UnitType.ARMY
                if province and province.province_type == ProvinceType.COAST:
                    unit_type = (
                        UnitType.FLEET if random.choice([True, False]) else UnitType.ARMY
                    )
                self.units[loc] = Unit(power, loc, unit_type)
            built[power] = selection
        if built:
            self.pending_builds = {}
        return built

    def total_supply_centers(self) -> int:
        return sum(1 for prov in self.board.values() if prov.is_supply_center)

    def determine_winner(self) -> Optional[Power]:
        total = self.total_supply_centers()
        if total == 0:
            return None
        majority = total // 2 + 1
        counts: Dict[Power, int] = {}
        for controller in self.supply_center_control.values():
            if controller is None:
                continue
            counts[controller] = counts.get(controller, 0) + 1
        for power, count in counts.items():
            if count >= majority:
                return power
        return None


__all__ = ["GameState"]
