from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Power(str):
    """Identifier for a player/power."""


class ProvinceType(Enum):
    LAND = auto()
    COAST = auto()
    SEA = auto()


@dataclass(frozen=True)
class Province:
    name: str
    neighbors: set[str]
    is_supply_center: bool = False
    home_power: Optional[Power] = None
    province_type: ProvinceType = ProvinceType.LAND


class Phase(Enum):
    SPRING = auto()
    SPRING_RETREAT = auto()
    FALL = auto()
    FALL_RETREAT = auto()


class OrderType(Enum):
    HOLD = auto()
    MOVE = auto()
    SUPPORT = auto()
    RETREAT = auto()


class UnitType(Enum):
    ARMY = auto()
    FLEET = auto()


@dataclass(frozen=True)
class Unit:
    power: Power
    loc: str  # province name
    unit_type: UnitType = UnitType.ARMY


@dataclass(frozen=True)
class Order:
    unit: Unit
    type: OrderType
    target: Optional[str] = None
    support_unit_loc: Optional[str] = None
    support_target: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - formatting only
        if self.type == OrderType.HOLD:
            return f"{self.unit.power} {self.unit.loc} H"
        if self.type == OrderType.MOVE:
            return f"{self.unit.power} {self.unit.loc} -> {self.target}"
        if self.type == OrderType.SUPPORT:
            if self.support_target:
                return (
                    f"{self.unit.power} {self.unit.loc} S "
                    f"{self.support_unit_loc} -> {self.support_target}"
                )
            return f"{self.unit.power} {self.unit.loc} S {self.support_unit_loc}"
        if self.type == OrderType.RETREAT:
            return f"{self.unit.power} {self.unit.loc} R -> {self.target}"
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
    if order.type == OrderType.RETREAT and order.target:
        return f"{unit.power} retreats from {unit.loc} to {order.target}."
    return f"{unit.power} issues an order."


__all__ = [
    "Power",
    "Province",
    "ProvinceType",
    "Phase",
    "OrderType",
    "UnitType",
    "Unit",
    "Order",
    "describe_order",
]
