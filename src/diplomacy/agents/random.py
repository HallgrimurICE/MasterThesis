from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ..orders import hold, move, retreat, support_hold, support_move
from ..state import GameState
from ..types import Order, Unit
from .base import Agent


class RandomAgent(Agent):
    """Agent that issues random legal orders."""

    def __init__(
        self,
        power,
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
        friendly_units = [
            unit for unit in state.units.values() if unit.power == self.power
        ]
        all_units: List[Unit] = list(state.units.values())
        legal_moves_map: Dict[str, List[str]] = {
            unit.loc: state.legal_moves_from(unit.loc) for unit in all_units
        }

        for unit in friendly_units:
            legal_moves = legal_moves_map.get(unit.loc, [])
            support_hold_targets = [
                nbr for nbr in state.graph.neighbors(unit.loc) if nbr in state.units
            ]
            support_move_options: List[Tuple[str, str]] = []
            for dest in state.graph.neighbors(unit.loc):
                for other_unit in all_units:
                    if other_unit.loc == unit.loc:
                        continue
                    other_moves = legal_moves_map.get(other_unit.loc, [])
                    if dest in other_moves:
                        support_move_options.append((other_unit.loc, dest))

            if (
                not legal_moves
                and not support_hold_targets
                and not support_move_options
            ):
                orders.append(hold(unit))
                continue

            if self._rng.random() < self.hold_probability:
                orders.append(hold(unit))
                continue

            action_pool: List[str] = []
            if legal_moves:
                action_pool.append("move")
            if support_hold_targets:
                action_pool.append("support_hold")
            if support_move_options:
                action_pool.append("support_move")

            if not action_pool:
                orders.append(hold(unit))
                continue

            action = self._rng.choice(action_pool)
            if action == "move":
                destination = self._rng.choice(legal_moves)
                orders.append(move(unit, destination))
            elif action == "support_hold":
                friend_loc = self._rng.choice(support_hold_targets)
                orders.append(support_hold(unit, friend_loc))
            else:  # support_move
                friend_from, friend_to = self._rng.choice(support_move_options)
                orders.append(support_move(unit, friend_from, friend_to))

        return orders

    def _plan_retreat_orders(
        self,
        state: "GameState",
        retreat_index: int,
    ) -> List[Order]:
        orders: List[Order] = []
        for origin, pending_unit in state.pending_retreats.items():
            if pending_unit.power != self.power:
                continue
            legal = state.legal_retreats_from(origin)
            if not legal:
                continue
            destination = self._rng.choice(legal)
            unit = Unit(self.power, origin)
            orders.append(retreat(unit, destination))
        return orders


__all__ = ["RandomAgent"]
