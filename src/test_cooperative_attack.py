import unittest

from game_engine import (
    Adjudicator,
    cooperative_attack_initial_state,
    move,
    hold,
    support_move,
)


class CooperativeAttackTest(unittest.TestCase):
    def test_solo_attack_bounces(self):
        state = cooperative_attack_initial_state()
        attacker = state.units["A"]
        ally = state.units["B"]
        defender = state.units["C"]

        orders = [
            move(attacker, "C"),
            hold(ally),
            hold(defender),
        ]

        next_state, resolution = Adjudicator(state).resolve(orders)

        self.assertNotIn("C", resolution.dislodged)
        self.assertIn(orders[0], resolution.failed)
        self.assertEqual(next_state.units["A"].power, attacker.power)
        self.assertEqual(next_state.units["C"].power, defender.power)

    def test_supported_attack_dislodges_defender(self):
        state = cooperative_attack_initial_state()
        attacker = state.units["A"]
        supporter = state.units["B"]
        defender = state.units["C"]

        orders = [
            move(attacker, "C"),
            support_move(supporter, "A", "C"),
            hold(defender),
        ]

        next_state, resolution = Adjudicator(state).resolve(orders)

        self.assertIn("C", resolution.dislodged)
        self.assertIn(orders[0], resolution.succeeded)
        self.assertIn(orders[1], resolution.succeeded)
        self.assertIn(orders[2], resolution.failed)
        self.assertEqual(next_state.units["C"].power, attacker.power)


if __name__ == "__main__":
    unittest.main()
