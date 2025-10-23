import unittest

from game_engine import (
    Adjudicator,
    Phase,
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

    def test_supply_centers_have_controller_field(self):
        state = cooperative_attack_initial_state()
        for loc, province in state.board.items():
            if province.is_supply_center:
                self.assertIn(loc, state.supply_center_control)
                self.assertIsNone(state.supply_center_control[loc])

    def test_control_updates_after_fall_resolution(self):
        state = cooperative_attack_initial_state()
        attacker = state.units["A"]
        supporter = state.units["B"]
        defender = state.units["C"]

        spring_orders = [
            move(attacker, "C"),
            support_move(supporter, "A", "C"),
            hold(defender),
        ]

        fall_candidate_state, _ = Adjudicator(state).resolve(spring_orders)
        self.assertEqual(fall_candidate_state.phase, Phase.FALL)
        self.assertIsNone(fall_candidate_state.supply_center_control["C"])

        fall_orders = [
            hold(fall_candidate_state.units["C"]),
            hold(fall_candidate_state.units["B"]),
        ]

        next_state, _ = Adjudicator(fall_candidate_state).resolve(fall_orders)
        self.assertEqual(next_state.phase, Phase.SPRING)
        self.assertEqual(next_state.supply_center_control["C"], attacker.power)
        self.assertEqual(next_state.supply_center_control["B"], supporter.power)


if __name__ == "__main__":
    unittest.main()
