from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from ..state import GameState
from ..types import Phase, Power, UnitType

from .observation import observation_utils, province_order

# Canonical power ordering used by the DeepMind agents.
_DM_POWER_ORDER = (
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
)
_POWER_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_DM_POWER_ORDER)}

_AREA_NAME_TO_ID = province_order.province_name_to_id(
    province_order.MapMDF.BICOASTAL_MAP
)
_PROVINCE_NAME_TO_ID = province_order.province_name_to_id(
    province_order.MapMDF.STANDARD_MAP
)

# The DeepMind map abbreviates a few sea zones differently than the engine
# board (e.g. English Channel is ``ECH``).  Provide aliases so we can map
# engine order/observation names to the ids used by the SL observation space.
_PROVINCE_ALIASES = {
    "ENG": "ECH",
    "LYO": "GOL",
    "BOT": "GOB",
}


def _normalise_name(name: str) -> str:
    return name.upper().strip()


def _area_id_for_location(loc: str) -> int:
    normalised = _normalise_name(loc)
    lookup = _PROVINCE_ALIASES.get(normalised, normalised)
    try:
        return _AREA_NAME_TO_ID[lookup]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown province/area name '{loc}'") from exc


def _power_to_index(power: Power, *, fallback: Iterable[Power] | None = None) -> int:
    normalised = _normalise_name(str(power))
    if normalised in _POWER_INDEX:
        return _POWER_INDEX[normalised]
    if fallback is not None:
        ordered = [
            _normalise_name(str(p))
            for p in sorted(set(fallback), key=str)
        ]
        if normalised in ordered:
            return ordered.index(normalised)
    raise KeyError(f"Unknown power '{power}' - please use standard power names")


def _encode_units(board: np.ndarray, state: GameState) -> None:
    board[:, observation_utils.OBSERVATION_UNIT_ABSENT] = 1
    for unit in state.units.values():
        area_id = _area_id_for_location(unit.loc)
        board[area_id, observation_utils.OBSERVATION_UNIT_ABSENT] = 0
        if unit.unit_type == UnitType.ARMY:
            board[area_id, observation_utils.OBSERVATION_UNIT_ARMY] = 1
        else:
            board[area_id, observation_utils.OBSERVATION_UNIT_FLEET] = 1
        try:
            power_idx = _power_to_index(unit.power, fallback=state.powers)
        except KeyError:
            continue
        board[
            area_id,
            observation_utils.OBSERVATION_UNIT_POWER_START + power_idx,
        ] = 1


def _encode_supply_centres(board: np.ndarray, control: Mapping[str, Power | None]) -> None:
    for province_name, controller in control.items():
        if controller is None:
            continue
        normalised = _normalise_name(province_name)
        province_id = _PROVINCE_NAME_TO_ID.get(normalised)
        if province_id is None:
            continue
        try:
            power_idx = _power_to_index(controller)
        except KeyError:
            continue
        start, num_areas = observation_utils.obs_index_start_and_num_areas(province_id)
        board[
            start : start + num_areas,
            observation_utils.OBSERVATION_SC_POWER_START + power_idx,
        ] = 1


def _phase_to_season(phase: Phase) -> observation_utils.Season:
    mapping = {
        Phase.SPRING: observation_utils.Season.SPRING_MOVES,
        Phase.SPRING_RETREAT: observation_utils.Season.SPRING_RETREATS,
        Phase.FALL: observation_utils.Season.AUTUMN_MOVES,
        Phase.FALL_RETREAT: observation_utils.Season.AUTUMN_RETREATS,
    }
    return mapping.get(phase, observation_utils.Season.SPRING_MOVES)


def _build_numbers(state: GameState) -> np.ndarray:
    builds = np.zeros((observation_utils.NUM_POWERS,), dtype=np.int32)
    for power, count in state.pending_builds.items():
        try:
            idx = _power_to_index(power, fallback=state.powers)
        except KeyError:
            continue
        builds[idx] = int(count)
    for power, count in state.pending_disbands.items():
        try:
            idx = _power_to_index(power, fallback=state.powers)
        except KeyError:
            continue
        builds[idx] = -int(count)
    return builds


def build_observation(
    state: GameState,
    last_actions: Sequence[int] | None = None,
) -> observation_utils.Observation:
    """Convert the engine GameState into DeepMind's Observation tuple."""

    board = np.zeros(
        observation_utils.OBSERVATION_BOARD_SHAPE, dtype=np.int8
    )
    _encode_units(board, state)
    _encode_supply_centres(board, state.supply_center_control)

    season = _phase_to_season(state.phase)
    build_numbers = _build_numbers(state)
    last_actions_arr = np.asarray(last_actions or (), dtype=np.int64)

    return observation_utils.Observation(
        season=season,
        board=board,
        build_numbers=build_numbers,
        last_actions=last_actions_arr,
    )