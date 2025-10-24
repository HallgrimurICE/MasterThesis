from __future__ import annotations

from typing import List, Set

from ..state import GameState
from ..types import Order, Unit


class Agent:
    """Base class for programmable agents that can issue orders each round."""

    def __init__(self, power):
        self.power = power
        self._round_index = 0
        self._retreat_index = 0

    def issue_orders(self, state: "GameState") -> List[Order]:
        """Return this power's orders for the current round."""

        if state.phase.name.endswith("RETREAT"):
            planned = self._plan_retreat_orders(state, self._retreat_index)
            self._retreat_index += 1
        else:
            planned = self._plan_orders(state, self._round_index)
            self._round_index += 1

        for order in planned:
            if order.unit.power != self.power:
                raise ValueError(
                    f"Agent for power {self.power} produced an order for {order.unit.power}."
                )
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

    def _plan_retreat_orders(self, state: "GameState", retreat_index: int) -> List[Order]:
        return []


__all__ = ["Agent"]
