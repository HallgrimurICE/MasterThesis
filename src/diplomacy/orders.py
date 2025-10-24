from __future__ import annotations

from .types import Order, OrderType, Unit


def hold(u: Unit) -> Order:
    return Order(unit=u, type=OrderType.HOLD)


def move(u: Unit, dest: str) -> Order:
    return Order(unit=u, type=OrderType.MOVE, target=dest)


def support_hold(u: Unit, friend_loc: str) -> Order:
    return Order(unit=u, type=OrderType.SUPPORT, support_unit_loc=friend_loc)


def support_move(u: Unit, friend_from: str, friend_to: str) -> Order:
    return Order(
        unit=u,
        type=OrderType.SUPPORT,
        support_unit_loc=friend_from,
        support_target=friend_to,
    )


def retreat(u: Unit, dest: str) -> Order:
    return Order(unit=u, type=OrderType.RETREAT, target=dest)


__all__ = ["hold", "move", "support_hold", "support_move", "retreat"]
