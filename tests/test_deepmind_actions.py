import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
POLICYTRAINING_DIR = PROJECT_ROOT / "policytraining"

for path in (SRC_DIR, POLICYTRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from diplomacy.deepmind import actions as dm_actions  # noqa: E402
from diplomacy.deepmind.actions import legal_actions_from_state  # noqa: E402
from diplomacy.maps import standard_board  # noqa: E402
from diplomacy.state import GameState  # noqa: E402
from diplomacy.types import Power, Unit, UnitType  # noqa: E402
from policytraining.environment import action_utils  # noqa: E402


class LegalActionsSupportTest(unittest.TestCase):

    def _actions_for_power(self, units, target_power):
        board = standard_board()
        state = GameState(board=board, units=units, powers=set(u.power for u in units.values()))
        legal_actions = legal_actions_from_state(state)
        ordered_powers = sorted(state.powers, key=str)
        power_index = ordered_powers.index(target_power)
        return [int(a) for a in legal_actions[power_index]]

    def test_support_hold_actions_are_present(self):
        france = Power("France")
        germany = Power("Germany")
        units = {
            "PAR": Unit(france, "PAR", UnitType.ARMY),
            "BUR": Unit(france, "BUR", UnitType.ARMY),
            "MUN": Unit(germany, "MUN", UnitType.ARMY),
        }
        france_actions = self._actions_for_power(units, france)
        supporter = dm_actions._province_id("PAR")
        supported = dm_actions._province_id("BUR")
        self.assertTrue(
            any(
                order == action_utils.SUPPORT_HOLD and src[0] == supporter and target[0] == supported
                for order, src, target, _ in map(action_utils.action_breakdown, france_actions)
            ),
            "Expected France to have a support-hold order for PAR supporting BUR",
        )

    def test_support_move_actions_are_present(self):
        france = Power("France")
        england = Power("England")
        units = {
            "BRE": Unit(france, "BRE", UnitType.FLEET),
            "PAR": Unit(france, "PAR", UnitType.ARMY),
            "LON": Unit(england, "LON", UnitType.ARMY),
        }
        france_actions = self._actions_for_power(units, france)
        supporter = dm_actions._province_id("BRE")
        supported = dm_actions._province_id("PAR")
        destination = dm_actions._province_id("PIC")
        self.assertTrue(
            any(
                order == action_utils.SUPPORT_MOVE_TO
                and src[0] == supporter
                and third[0] == supported
                and target[0] == destination
                for order, src, target, third in map(action_utils.action_breakdown, france_actions)
            ),
            "Expected France to have a support-move order for BRE supporting PAR-PIC",
        )


if __name__ == "__main__":
    unittest.main()
