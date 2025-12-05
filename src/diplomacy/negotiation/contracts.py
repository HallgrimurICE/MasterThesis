"""Negotiation contract data structures used by RSS + Peace deals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence

from ..types import Power


@dataclass(frozen=True)
class Contract:
    """Bilateral contract limiting the actions of two powers."""

    player_i: Power
    player_j: Power
    allowed_i: FrozenSet[int]
    allowed_j: FrozenSet[int]

    def allowed_for(self, power: Power) -> Optional[FrozenSet[int]]:
        if power == self.player_i:
            return self.allowed_i
        if power == self.player_j:
            return self.allowed_j
        return None


def _intersect_allowed(
    legal_actions: Sequence[int],
    restrictions: Sequence[FrozenSet[int]],
) -> Sequence[int]:
    """Return actions that satisfy all restrictions, falling back if empty."""

    if not restrictions:
        return legal_actions
    intersection = set(legal_actions)
    for subset in restrictions:
        intersection.intersection_update(subset)
    if not intersection:
        return legal_actions
    return [action for action in legal_actions if action in intersection]


def restrict_actions_for_power(
    power: Power,
    legal_actions: Sequence[int],
    contracts: Sequence[Contract],
) -> Sequence[int]:
    """Restrict ``legal_actions`` according to all contracts involving ``power``."""

    matching = [allowed for contract in contracts if (allowed := contract.allowed_for(power))]
    return _intersect_allowed(legal_actions, matching)


__all__ = ["Contract", "restrict_actions_for_power"]
