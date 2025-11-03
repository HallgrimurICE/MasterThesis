"""Tests for mapping ``GameState`` into DeepMind observations."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from diplomacy.environment import observation_utils as dm_utils
from diplomacy.environment import province_order

from diplomacy.deepmind import build_observation
from diplomacy.maps import standard_board
from diplomacy.state import GameState
from diplomacy.types import Phase, Power, Unit, UnitType


def _area_for_province(name: str) -> int:
    province_id = province_order.province_name_to_id()[name]
    area, _ = dm_utils.obs_index_start_and_num_areas(province_id)
    return area


def test_basic_observation_features() -> None:
    board = standard_board()
    units = {
        "PAR": Unit(Power("France"), "PAR", UnitType.ARMY),
        "ION": Unit(Power("Italy"), "ION", UnitType.FLEET),
    }
    powers = {Power("France"), Power("Italy")}
    state = GameState(board=board, units=units, powers=powers)

    obs = build_observation(state)

    assert obs.season == dm_utils.Season.SPRING_MOVES
    assert obs.board.shape == tuple(dm_utils.OBSERVATION_BOARD_SHAPE)
    np.testing.assert_array_equal(obs.build_numbers, np.zeros(dm_utils.NUM_POWERS, dtype=np.int32))
    assert obs.last_actions == []

    par_area = _area_for_province("PAR")
    assert obs.board[par_area, dm_utils.OBSERVATION_UNIT_ARMY] == 1.0
    assert obs.board[par_area, dm_utils.OBSERVATION_UNIT_ABSENT] == 0.0
    assert obs.board[par_area, dm_utils.OBSERVATION_UNIT_POWER_START + 2] == 1.0
    assert obs.board[par_area, dm_utils.OBSERVATION_SC_POWER_START + 2] == 1.0

    ion_area = _area_for_province("ION")
    assert obs.board[ion_area, dm_utils.OBSERVATION_UNIT_FLEET] == 1.0
    assert obs.board[ion_area, dm_utils.OBSERVATION_UNIT_ABSENT] == 0.0
    assert obs.board[ion_area, dm_utils.OBSERVATION_UNIT_POWER_START + 4] == 1.0


def test_dislodged_and_last_actions() -> None:
    board = standard_board()
    state = GameState(
        board=board,
        units={},
        powers={Power("Germany")},
        phase=Phase.SPRING_RETREAT,
    )
    state.pending_retreats["BER"] = Unit(Power("Germany"), "BER", UnitType.ARMY)

    obs = build_observation(state, last_actions=[42])

    assert obs.season == dm_utils.Season.SPRING_RETREATS
    assert obs.last_actions == [42]

    ber_area = _area_for_province("BER")
    assert obs.board[ber_area, dm_utils.OBSERVATION_DISLODGED_ARMY] == 1.0
    assert obs.board[ber_area, dm_utils.OBSERVATION_DISLODGED_FLEET + 1] == 0.0
    assert obs.board[ber_area, dm_utils.OBSERVATION_DISLODGED_START + 3] == 1.0
    assert obs.board[ber_area, dm_utils.OBSERVATION_UNIT_ABSENT] == 1.0
