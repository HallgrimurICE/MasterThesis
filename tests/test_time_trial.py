import random
import sys
import time
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
    from diplomacy.agents.random import RandomAgent
    from diplomacy.maps import cooperative_attack_initial_state
    from diplomacy.simulation import run_rounds_with_agents
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    if exc.name == "numpy":
        raise unittest.SkipTest("numpy is required for diplomacy tests") from exc
    raise


class CooperativeAttackTimingTest(unittest.TestCase):
    def test_single_round_average_time_under_threshold(self):
        """Ensure a small cooperative scenario resolves quickly on average."""

        iterations = 25
        start = time.perf_counter()

        for seed in range(iterations):
            initial_state = cooperative_attack_initial_state()
            agents = {
                power: RandomAgent(power, rng=random.Random(seed + idx))
                for idx, power in enumerate(sorted(initial_state.powers, key=str))
            }
            run_rounds_with_agents(initial_state, agents, rounds=1)

        elapsed = time.perf_counter() - start
        average_duration = elapsed / iterations

        self.assertLess(
            average_duration,
            0.25,
            msg=(
                "Cooperative attack round should resolve well under a quarter second on "
                f"average; observed {average_duration:.4f} seconds"
            ),
        )


if __name__ == "__main__":
    unittest.main()
