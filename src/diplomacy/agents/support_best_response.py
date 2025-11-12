"""Support-aware best-response agent utilities for Diplomacy.

The helpers in this module implement a light-weight approximation of a
support-capable sampled best response (SBR). Bundles of candidate orders are
constructed that include primary actions (move/hold) together with plausible
support orders from neighbouring friendly units. The bundled orders are then
scored via a sampled best-response evaluation procedure against opponent
profiles drawn from a stochastic base policy.

The heuristics deliberately favour supply centres and contested fronts so the
search focuses on strategically relevant moves while remaining computationally
tractable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import islice
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..adjudication import Adjudicator
from ..orders import hold, move, support_hold, support_move
from ..state import GameState
from ..types import Power, Unit

OrderBundle = Dict[Unit, str]


class BasePolicy(Protocol):
    """Protocol describing the policy interface required by the agent."""

    adjudicator: "AdjudicatorProtocol"

    def sample_joint_orders(
        self,
        state: GameState,
        *,
        exclude_power: Optional[Power] = None,
        temperature: float = 0.7,
    ) -> Dict[Unit, str]:
        ...


class AdjudicatorProtocol(Protocol):
    """Subset of the adjudicator interface required by the agent."""

    def legal_orders(self, state: GameState, unit: Unit) -> Sequence[str]:
        ...

    def apply_orders(self, state: GameState, joint_orders: Mapping[Unit, str]) -> GameState:
        ...


def unit_prefix(unit: Unit) -> str:
    """Return the canonical order prefix for the unit type."""

    return "F" if unit.unit_type.name == "FLEET" else "A"


def neighboring_friendly_units(state: GameState, player: Power, province: str) -> List[Unit]:
    """Return friendly units that border ``province``.

    A unit is considered neighbouring if it occupies a province directly
    adjacent (in the map graph) to ``province``.
    """

    neighbors: List[Unit] = []
    for neighbor_name in state.graph.neighbors(province):
        occupant = state.units.get(neighbor_name)
        if occupant is None:
            continue
        if occupant.power != player:
            continue
        neighbors.append(occupant)
    return neighbors


def is_supply_center(state: GameState, province: str) -> bool:
    """Return ``True`` iff the province is a supply centre."""

    province_data = state.board.get(province)
    return bool(province_data and province_data.is_supply_center)


def _enemy_neighbors(state: GameState, player: Power, province: str) -> int:
    return sum(
        1
        for neighbor in state.graph.neighbors(province)
        if neighbor in state.units and state.units[neighbor].power != player
    )


def simple_move_score(state: GameState, unit: Unit, dest: str) -> float:
    """Heuristic scoring for a candidate move.

    The scoring favours moves into supply centres, enemy-occupied provinces and
    provinces bordering enemy units. A small bonus is applied if the
    destination is more hotly contested than the origin.
    """

    score = 0.0
    if is_supply_center(state, dest):
        score += 2.0
    occupant = state.units.get(dest)
    if occupant and occupant.power != unit.power:
        score += 1.5
    enemy_neighbors = _enemy_neighbors(state, unit.power, dest)
    score += min(enemy_neighbors, 3) * 0.5
    origin_enemy_neighbors = _enemy_neighbors(state, unit.power, unit.loc)
    score += 0.2 * (enemy_neighbors - origin_enemy_neighbors)
    return score


def _supply_center_distance(state: GameState, province: str) -> int:
    if is_supply_center(state, province):
        return 0
    frontier = [(province, 0)]
    seen = {province}
    for current, distance in frontier:
        if is_supply_center(state, current):
            return distance
        for neighbor in state.graph.neighbors(current):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append((neighbor, distance + 1))
    return 999


def simple_support_score(state: GameState, supporter: Unit, target_province: str) -> float:
    """Rank potential supporters by proximity to supply-centre front lines."""

    score = 0.0
    if is_supply_center(state, target_province):
        score += 1.5
    score += max(0, 3 - min(3, _supply_center_distance(state, supporter.loc))) * 0.3
    score += _enemy_neighbors(state, supporter.power, supporter.loc) * 0.3
    return score


@dataclass(frozen=True)
class ParsedOrder:
    kind: str
    origin: str
    target: Optional[str] = None
    support_origin: Optional[str] = None


def _parse_order(order: str) -> ParsedOrder:
    parts = order.split()
    if len(parts) < 3:
        return ParsedOrder("UNKNOWN", parts[1] if len(parts) > 1 else "", None, None)
    origin = parts[1]
    if parts[2] == "H":
        return ParsedOrder("HOLD", origin, None, None)
    if parts[2] == "-" and len(parts) >= 4:
        return ParsedOrder("MOVE", origin, parts[3], None)
    if parts[2] == "S" and len(parts) >= 4:
        support_origin = parts[3]
        if len(parts) == 4:
            return ParsedOrder("SUPPORT_HOLD", origin, None, support_origin)
        if parts[4] == "-" and len(parts) >= 6:
            return ParsedOrder("SUPPORT_MOVE", origin, parts[5], support_origin)
        if parts[4] == "H":
            return ParsedOrder("SUPPORT_HOLD", origin, None, support_origin)
    return ParsedOrder("UNKNOWN", origin, None, None)


def _order_from_string(order: str, unit: Unit):
    """Convert a support-aware order string into an ``Order`` instance.

    Any malformed order strings gracefully degrade to a hold order so the
    adjudicator can still resolve the joint profile.
    """

    parsed = _parse_order(order)
    if parsed.kind == "MOVE" and parsed.target is not None:
        return move(unit, parsed.target)
    if parsed.kind == "SUPPORT_MOVE" and parsed.support_origin and parsed.target:
        return support_move(unit, parsed.support_origin, parsed.target)
    if parsed.kind == "SUPPORT_HOLD" and parsed.support_origin:
        return support_hold(unit, parsed.support_origin)
    return hold(unit)


def _format_support_hold(supporter: Unit, target_loc: str) -> str:
    return f"{unit_prefix(supporter)} {supporter.loc} S {target_loc} H"


def _format_support_move(supporter: Unit, mover: Unit, dest: str) -> str:
    return f"{unit_prefix(supporter)} {supporter.loc} S {mover.loc} - {dest}"


def _select_top(options: List[Tuple[str, float]], limit: int) -> List[str]:
    ranked = sorted(options, key=lambda item: item[1], reverse=True)
    return [order for order, _ in islice(ranked, limit)]


def propose_bundles(
    state: GameState,
    player: Power,
    base_policy: BasePolicy,
    *,
    k_moves: int = 5,
    m_supports: int = 2,
    include_holds: bool = True,
) -> List[OrderBundle]:
    """Construct bundled order candidates including support actions."""

    adjudicator = getattr(base_policy, "adjudicator", None)
    if adjudicator is None:
        raise ValueError("base_policy must expose an adjudicator attribute")

    friendly_units = [unit for unit in state.units.values() if unit.power == player]
    if not friendly_units:
        return []

    legal_map: Dict[Unit, Sequence[str]] = {
        unit: adjudicator.legal_orders(state, unit) for unit in friendly_units
    }

    candidate_orders: Dict[Unit, List[Tuple[str, float]]] = {}
    for unit in friendly_units:
        legal_orders = legal_map.get(unit, [])
        move_candidates: List[Tuple[str, float]] = []
        hold_candidate: Optional[Tuple[str, float]] = None
        for order in legal_orders:
            parsed = _parse_order(order)
            if parsed.kind == "HOLD":
                hold_candidate = (order, 0.0)
            elif parsed.kind == "MOVE" and parsed.target is not None:
                score = simple_move_score(state, unit, parsed.target)
                move_candidates.append((order, score))
        selected_moves = _select_top(
            move_candidates, max(0, k_moves - (1 if include_holds else 0))
        )
        unit_candidates: List[Tuple[str, float]] = [(order, 0.0) for order in selected_moves]
        if include_holds and hold_candidate is not None:
            unit_candidates.insert(0, hold_candidate)
        if not unit_candidates and hold_candidate is not None:
            unit_candidates.append(hold_candidate)
        if not unit_candidates and legal_orders:
            unit_candidates.append((legal_orders[0], 0.0))
        candidate_orders[unit] = unit_candidates

    if not candidate_orders:
        return []

    best_assignment: Dict[Unit, str] = {}
    for unit, options in candidate_orders.items():
        best_assignment[unit] = options[0][0]

    base_assignments: List[OrderBundle] = [dict(best_assignment)]
    for unit, options in candidate_orders.items():
        for order, _ in options[1:]:
            alt = dict(best_assignment)
            alt[unit] = order
            base_assignments.append(alt)

    bundles: List[OrderBundle] = []
    seen_keys: set[Tuple[str, ...]] = set()

    for assignment in base_assignments:
        support_map: Dict[Unit, List[Tuple[Unit, str, float]]] = {}
        for unit, order in assignment.items():
            parsed = _parse_order(order)
            if parsed.kind == "MOVE" and parsed.target is not None:
                supporters: List[Tuple[Unit, str, float]] = []
                for supporter in neighboring_friendly_units(state, player, parsed.target):
                    if supporter == unit:
                        continue
                    support_order = _format_support_move(supporter, unit, parsed.target)
                    if support_order not in legal_map.get(supporter, []):
                        continue
                    supporters.append(
                        (
                            supporter,
                            support_order,
                            simple_support_score(state, supporter, parsed.target),
                        )
                    )
                if supporters:
                    supporters = sorted(supporters, key=lambda item: item[2], reverse=True)
                    support_map[unit] = supporters[:m_supports]
            elif parsed.kind == "HOLD":
                supporters = []
                for supporter in neighboring_friendly_units(state, player, parsed.origin):
                    if supporter == unit:
                        continue
                    support_order = _format_support_hold(supporter, parsed.origin)
                    if support_order not in legal_map.get(supporter, []):
                        continue
                    supporters.append(
                        (
                            supporter,
                            support_order,
                            simple_support_score(state, supporter, parsed.origin),
                        )
                    )
                if supporters:
                    supporters = sorted(supporters, key=lambda item: item[2], reverse=True)
                    support_map[unit] = supporters[:m_supports]

        bundle_variants: List[OrderBundle] = [dict(assignment)]
        for supporters in support_map.values():
            new_variants: List[OrderBundle] = []
            for variant in bundle_variants:
                new_variants.append(variant)
                for supporter, support_order, _ in supporters:
                    current_order = variant.get(supporter)
                    if current_order == support_order:
                        continue
                    if current_order and _parse_order(current_order).kind.startswith("SUPPORT"):
                        continue
                    updated = dict(variant)
                    updated[supporter] = support_order
                    new_variants.append(updated)
                combined = dict(variant)
                conflict = False
                for supporter, support_order, _ in supporters:
                    current_order = combined.get(supporter)
                    if (
                        current_order
                        and current_order != support_order
                        and _parse_order(current_order).kind.startswith("SUPPORT")
                    ):
                        conflict = True
                        break
                    combined[supporter] = support_order
                if not conflict:
                    new_variants.append(combined)
            bundle_variants = new_variants

        for variant in bundle_variants:
            if len(variant) != len(friendly_units):
                continue
            legal = True
            for unit in friendly_units:
                order = variant.get(unit)
                if order is None:
                    legal = False
                    break
                if order not in legal_map.get(unit, []):
                    legal = False
                    break
            if not legal:
                continue
            key = tuple(sorted(f"{unit.power}:{unit.loc}:{order}" for unit, order in variant.items()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bundles.append(variant)

    return bundles


class StandardAdjudicatorAdapter:
    """Bridge between the string-based agent interface and the engine adjudicator."""

    def legal_orders(self, state: GameState, unit: Unit) -> List[str]:
        prefix = unit_prefix(unit)
        orders: set[str] = {f"{prefix} {unit.loc} H"}

        for dest in state.legal_moves_from(unit.loc):
            orders.add(f"{prefix} {unit.loc} - {dest}")

        for neighbor in state.graph.neighbors(unit.loc):
            occupant = state.units.get(neighbor)
            if occupant and occupant.power == unit.power:
                orders.add(_format_support_hold(unit, neighbor))

        for friend in state.units.values():
            if friend.power != unit.power or friend.loc == unit.loc:
                continue
            for dest in state.legal_moves_from(friend.loc):
                if dest in state.graph.neighbors(unit.loc):
                    orders.add(_format_support_move(unit, friend, dest))

        return sorted(orders)

    def apply_orders(self, state: GameState, joint_orders: Mapping[Unit, str]) -> GameState:
        orders = []
        for unit in state.units.values():
            order_str = joint_orders.get(unit)
            if order_str is None:
                orders.append(hold(unit))
                continue
            orders.append(_order_from_string(order_str, unit))

        next_state, _ = Adjudicator(state).resolve(orders)
        return next_state


def sbr_with_supports(
    state: GameState,
    player: Power,
    base_policy: BasePolicy,
    value_fn,
    *,
    B_base: int = 32,
    C_cand: int = 64,
    temperature_grid: Sequence[float] = (0.3, 0.7, 1.0),
) -> OrderBundle:
    """Sampled best response evaluation over support-aware bundles."""

    adjudicator = getattr(base_policy, "adjudicator", None)
    if adjudicator is None:
        raise ValueError("base_policy must expose an adjudicator attribute")

    opponent_samples: List[Dict[Unit, str]] = []
    if B_base <= 0:
        opponent_samples.append({})
    else:
        temps = list(temperature_grid) or [0.7]
        for i in range(B_base):
            tau = temps[i % len(temps)]
            opponent_samples.append(
                base_policy.sample_joint_orders(state, exclude_power=player, temperature=tau)
            )

    candidate_bundles = propose_bundles(state, player, base_policy)
    if not candidate_bundles:
        return {}
    if len(candidate_bundles) > C_cand:
        indices = random.sample(range(len(candidate_bundles)), C_cand)
        candidate_bundles = [candidate_bundles[idx] for idx in indices]

    best_score = float("-inf")
    best_bundle: Optional[OrderBundle] = None

    for bundle in candidate_bundles:
        total = 0.0
        for opponent_orders in opponent_samples:
            joint_orders: Dict[Unit, str] = dict(bundle)
            joint_orders.update(opponent_orders)
            next_state = adjudicator.apply_orders(state, joint_orders)
            total += value_fn(next_state, for_power=player)
        score = total / float(len(opponent_samples)) if opponent_samples else 0.0
        if score > best_score:
            best_score = score
            best_bundle = bundle

    return best_bundle or {}


class BestResponseAgent:
    """Agent wrapper that exposes the SBR search via a simple ``act`` method."""

    def __init__(
        self,
        *,
        base_policy: BasePolicy,
        value_fn,
        B_base: int = 32,
        C_cand: int = 64,
        temperature_grid: Sequence[float] = (0.3, 0.7, 1.0),
    ) -> None:
        self.base_policy = base_policy
        self.value_fn = value_fn
        self.B_base = B_base
        self.C_cand = C_cand
        self.temperature_grid = tuple(temperature_grid)

    def act(self, state: GameState, player: Power) -> OrderBundle:
        """Return the sampled best-response bundle for ``player``."""

        return sbr_with_supports(
            state,
            player,
            self.base_policy,
            self.value_fn,
            B_base=self.B_base,
            C_cand=self.C_cand,
            temperature_grid=self.temperature_grid,
        )


__all__ = [
    "BestResponseAgent",
    "propose_bundles",
    "sbr_with_supports",
    "StandardAdjudicatorAdapter",
    "neighboring_friendly_units",
    "is_supply_center",
    "simple_move_score",
    "simple_support_score",
]
