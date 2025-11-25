import random
import sys
from collections import OrderedDict
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
POLICYTRAINING_DIR = PROJECT_ROOT / "policytraining"

for path in (SRC_DIR, POLICYTRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:  # noqa: E402
    from diplomacy.agents.best_response import SampledBestResponsePolicy
    from diplomacy.agents.random import RandomAgent
    from diplomacy.maps import cooperative_attack_initial_state
    from diplomacy.orders import hold, move
    from diplomacy.simulation import run_rounds_with_agents
    from diplomacy.types import Power, Unit, UnitType
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    if exc.name == "numpy":
        raise unittest.SkipTest("numpy is required for diplomacy tests") from exc
    raise


class SampledBestResponsePolicyTest(unittest.TestCase):
    def test_rollout_limit_changes_candidate_combos(self):
        power = Power("TestPower")
        unit_a = Unit(power, "A", UnitType.ARMY)
        unit_b = Unit(power, "B", UnitType.ARMY)

        candidate_map = OrderedDict(
            [
                (unit_a.loc, [hold(unit_a), move(unit_a, unit_b.loc)]),
                (unit_b.loc, [hold(unit_b), move(unit_b, unit_a.loc)]),
            ]
        )

        low_policy = SampledBestResponsePolicy(rollout_limit=1, rng=random.Random(0))
        high_policy = SampledBestResponsePolicy(rollout_limit=3, rng=random.Random(0))

        low_combos = set(low_policy._enumerate_combos(candidate_map))
        high_combos = set(high_policy._enumerate_combos(candidate_map))
        baseline_combo = (candidate_map[unit_a.loc][0], candidate_map[unit_b.loc][0])

        self.assertEqual(low_combos, {baseline_combo})
        self.assertIn(baseline_combo, high_combos)
        self.assertEqual(len(high_combos), 3)


class RandomAgentSimulationTest(unittest.TestCase):
    def test_random_agents_play_round(self):
        initial_state = cooperative_attack_initial_state()
        agents = {
            power: RandomAgent(power, rng=random.Random(idx))
            for idx, power in enumerate(sorted(initial_state.powers, key=str))
        }

        states, titles, orders_history = run_rounds_with_agents(
            initial_state, agents, rounds=1
        )

        self.assertEqual(len(states), 2)
        self.assertEqual(len(titles), 2)
        self.assertEqual(len(orders_history), 1)

        first_round_orders = orders_history[0]
        for power in initial_state.powers:
            unit_count = sum(1 for unit in initial_state.units.values() if unit.power == power)
            power_orders = [order for order in first_round_orders if order.unit.power == power]
            self.assertEqual(len(power_orders), unit_count)


if __name__ == "__main__":
    unittest.main()
