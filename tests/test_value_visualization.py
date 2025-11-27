import random
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
POLICYTRAINING_DIR = PROJECT_ROOT / "policytraining"

for path in (SRC_DIR, POLICYTRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:  # noqa: E402
    import numpy as np
    import jax
    import jax.numpy as jnp
    from diplomacy.deepmind.build_observation import build_observation
    from diplomacy.deepmind.actions import legal_actions_from_state
    from diplomacy.maps import standard_board
    from diplomacy.state import GameState
    from diplomacy.types import Power, ProvinceType, Unit, UnitType
    from diplomacy.viz.mesh import MATPLOTLIB_AVAILABLE, NX_AVAILABLE, visualize_state_mesh
    from policytraining.network import config as net_config
    from policytraining.network import network_policy
    from policytraining.network import parameter_provider
    from policytraining.run_sl import make_sl_policy
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    if exc.name == "numpy":
        raise unittest.SkipTest("numpy is required for diplomacy tests") from exc
    raise


_STANDARD_POWERS = (
    Power("Austria"),
    Power("England"),
    Power("France"),
    Power("Germany"),
    Power("Italy"),
    Power("Russia"),
    Power("Turkey"),
)


class _RandomParameterProvider:
    """Generate random parameters/state for the Diplomacy network."""

    def __init__(self, network_cls, network_kwargs, rng_seed: int = 0):
        rng = jax.random.PRNGKey(rng_seed)
        params, net_state = network_cls.initial_inference_params_and_state(
            network_kwargs, rng, network_kwargs["num_players"]
        )
        self._params = params
        self._state = net_state
        self._step = jnp.array(0)

    def params_for_actor(self):
        return self._params, self._state, self._step


class ValueVisualizationTest(unittest.TestCase):
    def _random_standard_state(self, seed: int = 0) -> GameState:
        rng = random.Random(seed)
        board = standard_board()
        province_names = list(board)
        rng.shuffle(province_names)

        units = {}
        for power, loc in zip(_STANDARD_POWERS, province_names):
            province = board[loc]
            if province.province_type == ProvinceType.SEA:
                unit_type = UnitType.FLEET
            elif province.province_type == ProvinceType.COAST:
                unit_type = rng.choice([UnitType.ARMY, UnitType.FLEET])
            else:
                unit_type = UnitType.ARMY
            units[loc] = Unit(power=power, loc=loc, unit_type=unit_type)

        return GameState(board=board, units=units, powers=set(_STANDARD_POWERS))

    def _load_policy(self):
        params_path = Path("policytraining/data/sl_params.npz")
        if params_path.is_file():
            return make_sl_policy(str(params_path))

        cfg = net_config.get_config()
        network_cls = cfg.network_class
        # Use training mode when seeding random parameters so that batch norm
        # statistics are initialized. The handler below will use the same
        # configuration for inference.
        network_kwargs = dict(cfg.network_kwargs, is_training=True)
        provider = _RandomParameterProvider(network_cls, network_kwargs)
        handler = parameter_provider.SequenceNetworkHandler(
            network_cls=network_cls,
            network_config=network_kwargs,
            rng_seed=0,
            parameter_provider=provider,
        )
        handler.reset()
        return network_policy.Policy(
            network_handler=handler,
            num_players=network_kwargs["num_players"],
            temperature=0.1,
            calculate_all_policies=False,
        )

    def test_random_state_value_and_visualization(self):
        state = self._random_standard_state(seed=123)
        observation = build_observation(state, last_actions=[])
        legal_actions = legal_actions_from_state(state)
        self.assertEqual(len(legal_actions), len(_STANDARD_POWERS))

        policy = self._load_policy()
        slots_list = list(range(len(legal_actions)))
        _, info = policy.actions(
            slots_list=slots_list,
            observation=observation,
            legal_actions=legal_actions,
        )

        values = np.asarray(info["values"], dtype=float)
        self.assertEqual(values.shape[0], len(_STANDARD_POWERS))
        self.assertAlmostEqual(values.sum(), 1.0, places=5)

        if not (MATPLOTLIB_AVAILABLE and NX_AVAILABLE):
            self.skipTest("Visualization dependencies are not installed")

        import matplotlib

        matplotlib.use("Agg", force=True)

        with patch("matplotlib.pyplot.show"):
            visualize_state_mesh(state, title="Random standard-board state")


if __name__ == "__main__":
    unittest.main()
