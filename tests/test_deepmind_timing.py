import os
import unittest
from pathlib import Path

from src.diplomacy.demo import deepmind_single_move_latency


class DeepMindAgentTimingTest(unittest.TestCase):
    def test_single_move_latency_with_minimal_search(self):
        """Measure how quickly the DeepMind SL agent returns one move at k=1, N=1."""

        # Your actual params file path:
        params_path = Path("C:/Users/hat/Documents/Thesis/MasterThesis/data/fppi2_params.npz")

        if not params_path.is_file():
            self.skipTest(
                "DeepMind SL parameters not found. "
                "Expected at C:/Users/hat/Documents/Thesis/MasterThesis/data/fppi2_params.npz"
            )

        duration = deepmind_single_move_latency(
            weights_path=params_path,
            k_candidates=1,
            action_rollouts=2,
            seed=123,
        )

        self.assertLess(
            duration,
            15.0,
            msg=(
                "DeepMind SL agent should generate orders quickly when k=1, N=1; "
                f"observed {duration:.3f}s"
            ),
        )


if __name__ == "__main__":
    # Run the test 10 times and print the average duration
    from statistics import mean

    # Set up like the test function does
    params_path = Path("C:/Users/hat/Documents/Thesis/MasterThesis/data/fppi2_params.npz")
    if not params_path.is_file():
        print("DeepMind SL parameters not found at", params_path)
    else:
        durations = []
        for i in range(10):
            duration = deepmind_single_move_latency(
                weights_path=params_path,
                k_candidates=3,
                action_rollouts=2,
                seed=123 + i,
            )
            print(f"Run {i+1}: {duration:.4f}s")
            durations.append(duration)
        avg = mean(durations)
        print(f"Average duration over 10 runs: {avg:.4f}s")
