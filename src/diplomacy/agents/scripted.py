from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

from ..orders import hold, move, retreat, support_move, support_hold
from ..state import GameState
from ..types import Order, Unit
from .base import Agent

Directive = Union[str, Order, None, Callable[[Unit, GameState], Order]]


class ScriptedAgent(Agent):
    """Agent whose behaviour is programmed via a per-round script."""

    def __init__(
        self,
        power,
        script: Dict[int, Dict[str, Directive]],
        *,
        retreat_script: Optional[Dict[int, Dict[str, Directive]]] = None,
    ):
        super().__init__(power)
        self.script = script
        self.retreat_script = retreat_script or {}

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
                upper = directive.upper()
                if upper in {"H", "HOLD"}:
                    orders.append(hold(unit))
                    continue
                if directive in state.legal_moves_from(unit.loc):
                    orders.append(move(unit, directive))
                    continue

            orders.append(hold(unit))

        return orders

    def _plan_retreat_orders(
        self,
        state: "GameState",
        retreat_index: int,
    ) -> List[Order]:
        orders: List[Order] = []
        planned_orders = self.retreat_script.get(retreat_index, {})
        for origin, pending_unit in state.pending_retreats.items():
            if pending_unit.power != self.power:
                continue

            directive = planned_orders.get(origin)
            unit = Unit(self.power, origin)

            if callable(directive):
                orders.append(directive(unit, state))
                continue
            if isinstance(directive, Order):
                orders.append(directive)
                continue

            if directive is None:
                continue

            if isinstance(directive, str):
                upper = directive.upper()
                if upper in {"D", "DISBAND"}:
                    continue
                orders.append(retreat(unit, directive))
                continue

        return orders


__all__ = ["ScriptedAgent", "Directive"]
