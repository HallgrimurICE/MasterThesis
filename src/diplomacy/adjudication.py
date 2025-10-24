from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .orders import hold
from .state import GameState
from .types import Order, OrderType, Phase, Power, Unit


@dataclass
class Resolution:
    succeeded: Set[Order] = field(default_factory=set)
    failed: Set[Order] = field(default_factory=set)
    dislodged: Set[str] = field(default_factory=set)  # provinces where unit was dislodged
    dislodged_units: Dict[str, Unit] = field(default_factory=dict)
    dislodged_by: Dict[str, Set[str]] = field(default_factory=dict)
    contested: Set[str] = field(default_factory=set)
    disbanded_retreats: Set[str] = field(default_factory=set)
    winner: Optional[Power] = None
    auto_disbands: Dict[Power, List[str]] = field(default_factory=dict)


class Adjudicator:
    def __init__(self, state: GameState):
        self.state = state

    def resolve(self, orders: Iterable[Order]) -> Tuple[GameState, Resolution]:
        if self.state.phase in {Phase.SPRING_RETREAT, Phase.FALL_RETREAT}:
            return self._resolve_retreat_phase(list(orders))
        return self._resolve_movement_phase(list(orders))

    def _resolve_movement_phase(self, orders: List[Order]) -> Tuple[GameState, Resolution]:
        by_loc: Dict[str, Order] = {o.unit.loc: o for o in orders}
        for loc, unit in self.state.units.items():
            if loc not in by_loc:
                by_loc[loc] = hold(unit)
        orders = list(by_loc.values())

        attacks_to: Dict[str, List[Order]] = {}
        supports: List[Order] = []
        for o in orders:
            if o.type == OrderType.MOVE:
                if o.target not in self.state.legal_moves_from(o.unit.loc):
                    continue
                attacks_to.setdefault(o.target, []).append(o)
            elif o.type == OrderType.SUPPORT:
                supports.append(o)

        valid_supports: Set[Order] = set()
        for s in supports:
            sup_prov = s.unit.loc
            attackers = attacks_to.get(sup_prov, [])
            cut = False
            for atk in attackers:
                if s.support_target is not None:
                    if not (atk.unit.loc == s.support_unit_loc and atk.target == s.support_target):
                        cut = True
                        break
                else:
                    if not (
                        atk.unit.loc == s.support_unit_loc
                        and atk.type == OrderType.MOVE
                        and atk.target == sup_prov
                    ):
                        cut = True
                        break
            if not cut:
                valid_supports.add(s)

        hold_strength: Dict[str, int] = {loc: 1 for loc in self.state.units.keys()}
        attack_strength: Dict[Tuple[str, str], int] = {}

        for s in valid_supports:
            if s.support_target is None and s.support_unit_loc in self.state.units:
                hold_strength[s.support_unit_loc] = hold_strength.get(s.support_unit_loc, 1) + 1

        for o in orders:
            if o.type == OrderType.MOVE and o.target in self.state.graph[o.unit.loc]:
                attack_strength[(o.unit.loc, o.target)] = 1
        for s in valid_supports:
            if s.support_target is not None:
                key = (s.support_unit_loc, s.support_target)
                if key in attack_strength:
                    attack_strength[key] += 1

        resolution = Resolution()
        new_units: Dict[str, Unit] = dict(self.state.units)
        dislodged: Set[str] = set()
        dislodged_units: Dict[str, Unit] = {}
        dislodged_by: Dict[str, Set[str]] = {}
        contested: Set[str] = set()
        successful_moves: List[Tuple[str, str, Unit]] = []

        winners_by_dest: Dict[str, List[Tuple[Order, int]]] = {}
        for dest, incoming in attacks_to.items():
            candidates: List[Tuple[Order, int]] = []
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
                contested.add(dest)
                continue
            winner, w_s = winners[0]
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
                    if dest_unit is not None:
                        dislodged_units[dest] = dest_unit
                        dislodged_by.setdefault(dest, set()).add(winner.unit.loc)
                    successful_moves.append((winner.unit.loc, dest, winner.unit))
                elif atk_ba > atk_ab:
                    continue
                else:
                    continue
            else:
                defense = hold_strength.get(dest, 0)
                if dest_unit is not None and dest_unit.power == winner.unit.power:
                    continue
                if w_s > defense:
                    if dest_unit is not None:
                        dislodged.add(dest)
                        dislodged_units[dest] = dest_unit
                        dislodged_by.setdefault(dest, set()).add(winner.unit.loc)
                    successful_moves.append((winner.unit.loc, dest, winner.unit))
                else:
                    continue

        resolution.dislodged = dislodged
        resolution.dislodged_units = dislodged_units
        resolution.dislodged_by = dislodged_by
        resolution.contested = contested

        for origin, dest, unit in successful_moves:
            if origin in dislodged:
                continue
            new_units.pop(origin, None)
            new_units[dest] = unit

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
            elif o.type == OrderType.SUPPORT:
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
            else:
                resolution.failed.add(o)

        normalized_units: Dict[str, Unit] = {
            loc: Unit(unit.power, loc) for loc, unit in new_units.items()
        }
        next_state = self.state.copy()
        next_state.units = normalized_units
        prev_phase = self.state.phase
        next_state.pending_retreats = {
            loc: Unit(unit.power, loc) for loc, unit in dislodged_units.items()
        }
        next_state.retreat_forbidden = {
            loc: set(origins) for loc, origins in dislodged_by.items()
        }
        next_state.contested_provinces = contested
        next_state.supply_update_due = prev_phase == Phase.FALL

        if dislodged:
            next_state.phase = (
                Phase.SPRING_RETREAT if prev_phase == Phase.SPRING else Phase.FALL_RETREAT
            )
        else:
            next_phase = Phase.FALL if prev_phase == Phase.SPRING else Phase.SPRING
            next_state.phase = next_phase
            if next_state.supply_update_due:
                next_state.update_supply_center_control(prev_phase=Phase.FALL)
                next_state.supply_update_due = False
                winner = next_state.determine_winner()
                next_state.winner = winner
                resolution.winner = winner
                next_state.update_pending_disbands()
                disbanded = next_state.auto_disband()
                if disbanded:
                    resolution.auto_disbands = disbanded

        return next_state, resolution

    def _resolve_retreat_phase(self, orders: List[Order]) -> Tuple[GameState, Resolution]:
        pending = dict(self.state.pending_retreats)
        resolution = Resolution()
        next_state = self.state.copy()
        next_state.units = dict(self.state.units)

        orders_by_origin: Dict[str, Order] = {}
        for order in orders:
            if order.type != OrderType.RETREAT:
                continue
            origin = order.unit.loc
            if origin not in pending:
                continue
            if order.unit.power != pending[origin].power:
                continue
            if origin in orders_by_origin:
                continue
            orders_by_origin[origin] = order

        legal_cache: Dict[str, Set[str]] = {
            loc: set(self.state.legal_retreats_from(loc)) for loc in pending
        }

        destinations: Dict[str, List[Order]] = {}
        for origin, order in orders_by_origin.items():
            legal_targets = legal_cache.get(origin, set())
            if order.target is None or order.target not in legal_targets:
                resolution.failed.add(order)
                resolution.disbanded_retreats.add(origin)
                print(
                    f"[Retreat] {order.unit.power} unit in {origin} fails to retreat "
                    f"to {order.target}; unit disbands."
                )
                continue
            destinations.setdefault(order.target, []).append(order)

        for origin in pending:
            if origin not in orders_by_origin:
                resolution.disbanded_retreats.add(origin)
                unit = pending[origin]
                print(f"[Retreat] {unit.power} unit in {origin} disbands (no retreat order).")

        for dest, competing_orders in destinations.items():
            if len(competing_orders) > 1:
                for order in competing_orders:
                    resolution.failed.add(order)
                    resolution.disbanded_retreats.add(order.unit.loc)
                    print(
                        f"[Retreat] {order.unit.power} unit in {order.unit.loc} fails to "
                        f"retreat to {order.target} due to standoff; unit disbands."
                    )
                continue
            order = competing_orders[0]
            resolution.succeeded.add(order)
            unit = pending[order.unit.loc]
            next_state.units[dest] = Unit(unit.power, dest)
            print(f"[Retreat] {unit.power} unit retreats from {order.unit.loc} to {dest}.")

        next_state.pending_retreats = {}
        next_state.retreat_forbidden = {}
        next_state.contested_provinces = set()

        if self.state.phase == Phase.SPRING_RETREAT:
            next_state.phase = Phase.FALL
            next_state.supply_update_due = self.state.supply_update_due
        else:
            next_state.phase = Phase.SPRING
            if self.state.supply_update_due:
                next_state.update_supply_center_control(prev_phase=Phase.FALL)
                next_state.supply_update_due = False
                winner = next_state.determine_winner()
                next_state.winner = winner
                resolution.winner = winner
                next_state.update_pending_disbands()
                disbanded = next_state.auto_disband()
                if disbanded:
                    resolution.auto_disbands = disbanded
            else:
                next_state.supply_update_due = False

        return next_state, resolution


__all__ = ["Resolution", "Adjudicator"]
