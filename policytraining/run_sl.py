# run_sl.py

import numpy as np

from policytraining.network import config as net_config
from policytraining.network import network_policy
from policytraining.network import parameter_provider
from policytraining.network.npz_parameter_provider import NpzParameterProvider  # <-- you created this


def make_sl_policy(sl_params_path: str, rng_seed: int = 0):
    """
    Create a Policy object using pretrained SL parameters from an .npz file.
    This does NOT depend on any particular environment.
    """
    # 1) Get the original network config (class + kwargs)
    cfg = net_config.get_config()          # ml_collections.ConfigDict
    net_cls = cfg.network_class            # e.g. network.Network
    net_kwargs = dict(cfg.network_kwargs)  # layer sizes etc.

    # 2) Use your NPZ-based parameter provider
    provider = NpzParameterProvider(sl_params_path)

    # 3) Build a SequenceNetworkHandler with the same network config
    handler = parameter_provider.SequenceNetworkHandler(
        network_cls=net_cls,
        network_config=net_kwargs,
        rng_seed=rng_seed,
        parameter_provider=provider,
    )

    # Initialise internal state / params from the provider
    handler.reset()

    # 4) Wrap into a Policy, which gives you `.actions(...)`
    num_players = net_kwargs["num_players"]
    temperature = 0.1  # typical eval temperature in their work

    policy = network_policy.Policy(
        network_handler=handler,
        num_players=num_players,
        temperature=temperature,
        calculate_all_policies=False,
    )
    return policy


if __name__ == "__main__":
    # Simple smoke test: just try to construct the policy.
    sl_path = "C:/Users/hat/Documents/Thesis/MasterThesis/policytraining/data/sl_params.npz"  # adjust path if needed
    policy = make_sl_policy(sl_path)
    print("SL policy created successfully:", type(policy))
