import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
POLICYTRAINING_DIR = PROJECT_ROOT / "policytraining"

for path in (SRC_DIR, POLICYTRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:  # noqa: E402
    from diplomacy.agents.sl_agent import DeepMindSlAgent
    from diplomacy.maps import cooperative_attack_initial_state
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - import guard
    missing = getattr(exc, "name", None) or "dependency"
    raise unittest.SkipTest(f"Missing dependency for DeepMind tests: {missing}") from exc


class DeepMindAgentTimingTest(unittest.TestCase):
    def test_single_move_latency_with_minimal_search(self):
        """Measure how quickly the DeepMind SL agent returns one move at k=1, N=1."""

        params_path = Path(os.environ.get("SL_PARAMS_PATH", ""))
        if not params_path:
            params_path = PROJECT_ROOT / "policytraining" / "data" / "sl_params.npz"

        if not params_path.is_file():
            self.skipTest(
                "DeepMind SL parameters not found. Set SL_PARAMS_PATH to sl_params.npz to run timing."
            )

        state = cooperative_attack_initial_state()
        power = sorted(state.powers, key=str)[0]

        agent = DeepMindSlAgent(
            power=power,
            sl_params_path=str(params_path),
            k_candidates=1,
            action_rollouts=1,
        )

        start = time.perf_counter()
        orders = agent._plan_orders(state, round_index=0)
        duration = time.perf_counter() - start

        self.assertTrue(orders, "Agent should return at least one order")
        self.assertLess(
            duration,
            15.0,
            msg=(
                "DeepMind SL agent should generate orders quickly when k=1, N=1; "
                f"observed {duration:.3f}s"
            ),
        )


if __name__ == "__main__":
    unittest.main()
