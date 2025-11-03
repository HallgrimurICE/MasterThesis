from diplomacy.deepmind import build_observation
from diplomacy.maps import standard_board
from diplomacy.state import GameState
from diplomacy.types import Power, Unit, UnitType


if __name__ == "__main__":

    board = standard_board()
    units = {
        "PAR": Unit(Power("France"), "PAR", UnitType.ARMY),
        "ION": Unit(Power("Italy"), "ION", UnitType.FLEET),
    }
    state = GameState(board=board, units=units, powers={Power("France"), Power("Italy")})
    obs = build_observation(state)

    print(obs)

