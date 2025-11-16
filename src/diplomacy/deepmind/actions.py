from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..state import GameState
from ..types import Order, OrderType, Power

from .observation import NUM_ACTIONS, province_order


def _ordered_powers(state: GameState) -> List[Power]:
    """Define a consistent ordering of powers (players)."""
    return sorted(state.powers, key=str)


def legal_actions_from_state(state: GameState) -> Sequence[np.ndarray]:
    """
    Return legal action indices in DeepMind's action space for each player.

    First milestone: only needed so the network can pick *some* actions.

    Returns:
        list of np.ndarray, length = num_players.
        Each array is the list of integer indices into the global DM action space
        that are legal for that player in the current phase.
    """

    powers = _ordered_powers(state)
    num_players = len(powers)

    # DeepMind's action space is "all possible orders for all units
    # in all areas", enumerated as 0..N-1 using a fixed ordering over
    # provinces and order templates.
    #
    # You will have to build an `all_actions` list (or reuse one from the DM repo)
    # and then select those that are compatible with the current GameState.
    #
    # For a *minimal* running version, you can do something much simpler:
    #   - For each power, return "all actions" (no masking).
    #   - This is suboptimal for performance but lets you test that the network runs.
    #
    all_action_indices = np.arange(NUM_ACTIONS, dtype=np.int32)

    legal_actions: List[np.ndarray] = []
    for _p in powers:
        # SUPER ROUGH: treat every action as legal for now.
        # Later, restrict based on which units/powers are on the board.
        legal_actions.append(all_action_indices)

    return legal_actions


def decode_actions_to_orders(
    state: GameState,
    power: Power,
    action_indices: Sequence[int],
) -> List[Order]:
    """Convert policy-chosen action indices into engine Order objects.

    The full DeepMind action decoding logic is quite involved.  For now we
    conservatively map every action to a HOLD order for each of the power's
    units so that callers receive a syntactically valid order list.
    """

    del action_indices  # Currently unused in the stub implementation.
    orders: List[Order] = []
    for unit in state.units.values():
        if unit.power != power:
            continue
        orders.append(Order(unit=unit, type=OrderType.HOLD))
    return orders