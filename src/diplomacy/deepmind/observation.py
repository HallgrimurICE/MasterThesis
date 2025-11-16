from __future__ import annotations

from policytraining.environment import observation_utils as _observation_utils
from policytraining.environment import province_order as _province_order
from policytraining.environment import action_list as _action_list

observation_utils = _observation_utils
province_order = _province_order
POSSIBLE_ACTIONS = _action_list.POSSIBLE_ACTIONS
NUM_ACTIONS = int(POSSIBLE_ACTIONS.shape[0])

__all__ = [
    "NUM_ACTIONS",
    "POSSIBLE_ACTIONS",
    "observation_utils",
    "province_order",
]