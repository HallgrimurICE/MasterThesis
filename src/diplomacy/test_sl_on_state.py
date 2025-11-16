# test_sl_on_state.py

from pathlib import Path

from diplomacy.maps import standard_board
from diplomacy.state import GameState
from diplomacy.types import Power, Unit, UnitType

from diplomacy.deepmind import build_observation, legal_actions_from_state
from policytraining.run_sl import make_sl_policy


def main():
    # 1) Build a simple test GameState
    board = standard_board()
    units = {
        "PAR": Unit(Power("France"), "PAR", UnitType.ARMY),
        "ION": Unit(Power("Italy"), "ION", UnitType.FLEET),
    }
    powers = {Power("France"), Power("Italy")}
    state = GameState(board=board, units=units, powers=powers)

    # 2) DeepMind observation + legal actions
    obs = build_observation(state, last_actions=[42])
    legal_actions = legal_actions_from_state(state)

    # 3) Load SL policy
    policy = make_sl_policy("data/sl_params.npz")

    # 4) Ask for actions for all players
    slots_list = list(range(len(legal_actions)))
    actions, info = policy.actions(
        slots_list=slots_list,
        observation=obs,
        legal_actions=legal_actions,
    )

    print("Slots list:", slots_list)
    print("Chosen action indices per slot:", actions)
    print("Values:", info["values"])
    print("Policy shape:", info["policy"].shape)


if __name__ == "__main__":
    main()
