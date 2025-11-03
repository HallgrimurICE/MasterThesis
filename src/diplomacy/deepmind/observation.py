"""Conversion helpers to build DeepMind-style observations from ``GameState``."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised indirectly in import-time tests
    from diplomacy.environment import observation_utils as dm_utils
    from diplomacy.environment import province_order
except ModuleNotFoundError:  # pragma: no cover - dependency hinting
    # ``diplomacy-main`` (DeepMind's release) lives alongside this project in
    # the repository. When the package is not installed into the environment we
    # fall back to loading the modules directly from that sibling checkout so
    # users can run the helpers without tweaking ``PYTHONPATH`` manually.
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    _DM_ROOT = Path(__file__).resolve().parents[3] / "diplomacy-main"
    _DM_ENVIRONMENT = _DM_ROOT / "environment"
    if not _DM_ENVIRONMENT.exists():
        raise

    def _load_dm_module(module_name: str, filename: str) -> ModuleType:
        module_path = _DM_ENVIRONMENT / filename
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ModuleNotFoundError(f"Unable to load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module

    # Ensure ``diplomacy.environment`` package placeholder exists before loading submodules.
    env_module = sys.modules.get("diplomacy.environment")
    if env_module is None:
        env_module = ModuleType("diplomacy.environment")
        env_module.__path__ = [str(_DM_ENVIRONMENT)]  # type: ignore[attr-defined]
        sys.modules["diplomacy.environment"] = env_module
    else:  # pragma: no cover - defensive cache reuse
        existing_path = getattr(env_module, "__path__", [])
        path_str = str(_DM_ENVIRONMENT)
        if path_str not in existing_path:
            env_module.__path__ = list(existing_path) + [path_str]  # type: ignore[attr-defined]

    dm_utils = _load_dm_module(
        "diplomacy.environment.observation_utils", "observation_utils.py"
    )
    province_order = _load_dm_module(
        "diplomacy.environment.province_order", "province_order.py"
    )

    # Wire utilities onto the namespace so standard imports succeed.
    env_module.observation_utils = dm_utils
    env_module.province_order = province_order
    diplomacy_pkg = sys.modules.get("diplomacy")
    if diplomacy_pkg is not None:
        setattr(diplomacy_pkg, "environment", env_module)

from ..state import GameState
from ..types import Phase, Power, Unit, UnitType

__all__ = ["build_observation", "dm_utils", "province_order"]


_POWER_ORDER = [
    "Austria",
    "England",
    "France",
    "Germany",
    "Italy",
    "Russia",
    "Turkey",
]
_POWER_TO_INDEX: Mapping[str, int] = {power: idx for idx, power in enumerate(_POWER_ORDER)}

_PHASE_TO_SEASON: Mapping[Phase, dm_utils.Season] = {
    Phase.SPRING: dm_utils.Season.SPRING_MOVES,
    Phase.SPRING_RETREAT: dm_utils.Season.SPRING_RETREATS,
    Phase.FALL: dm_utils.Season.AUTUMN_MOVES,
    Phase.FALL_RETREAT: dm_utils.Season.AUTUMN_RETREATS,
}

_PROVINCE_NAME_TO_ID = province_order.province_name_to_id()
_NO_UNIT_OWNER_INDEX = dm_utils.OBSERVATION_UNIT_POWER_START + dm_utils.NUM_POWERS
_DISLODGED_NONE_FLAG_INDEX = dm_utils.OBSERVATION_DISLODGED_FLEET + 1
_DISLODGED_NO_OWNER_INDEX = dm_utils.OBSERVATION_DISLODGED_START + dm_utils.NUM_POWERS
_AREA_TYPE_OFFSET = dm_utils.OBSERVATION_SC_POWER_START - 3
_SC_NO_OWNER_INDEX = dm_utils.OBSERVATION_SC_POWER_START + dm_utils.NUM_POWERS

def build_observation(
    game_state: GameState,
    *,
    last_actions: Optional[Sequence[int]] = None,
) -> dm_utils.Observation:
    """Return a DeepMind observation that mirrors ``game_state``.

    Args:
        game_state: Source game state from the lightweight engine.
        last_actions: Optional flat list of the last phase's DM action integers.

    Returns:
        ``dm_utils.Observation`` populated with features expected by the
        DeepMind policy stack.
    """

    season = _PHASE_TO_SEASON.get(game_state.phase)
    if season is None:
        raise ValueError(f"Unsupported phase for DeepMind observations: {game_state.phase!r}")

    board = _initial_board()
    _populate_units(board, game_state.units.values())
    _populate_dislodged_units(board, game_state.pending_retreats)
    _populate_supply_centers(board, game_state)

    build_numbers = _build_numbers(game_state)
    actions = list(last_actions) if last_actions is not None else []

    return dm_utils.Observation(
        season=season,
        board=board,
        build_numbers=build_numbers,
        last_actions=actions,
    )


def _initial_board() -> np.ndarray:
    board = np.zeros(dm_utils.OBSERVATION_BOARD_SHAPE, dtype=np.float32)
    board[:, dm_utils.OBSERVATION_UNIT_ABSENT] = 1.0
    board[:, _NO_UNIT_OWNER_INDEX] = 1.0
    board[:, _DISLODGED_NONE_FLAG_INDEX] = 1.0
    board[:, _DISLODGED_NO_OWNER_INDEX] = 1.0
    board[:, _SC_NO_OWNER_INDEX] = 1.0

    for area in range(dm_utils.NUM_AREAS):
        province_id, area_index = dm_utils.province_id_and_area_index(area)
        province_type = dm_utils.province_type_from_id(province_id)
        if province_type == dm_utils.ProvinceType.SEA:
            board[area, _AREA_TYPE_OFFSET + 1] = 1.0
        elif province_type == dm_utils.ProvinceType.BICOASTAL and area_index > 0:
            board[area, _AREA_TYPE_OFFSET + 2] = 1.0
        else:
            board[area, _AREA_TYPE_OFFSET + 0] = 1.0
    return board


def _populate_units(board: np.ndarray, units: Iterable[Unit]) -> None:
    for unit in units:
        province_id = _province_id(unit.loc)
        for area in _areas_for_unit(province_id, unit.unit_type):
            _set_unit_presence(board, area, unit.unit_type, unit.power)


def _populate_dislodged_units(
    board: np.ndarray,
    pending_retreats: Mapping[str, Unit],
) -> None:
    for province, unit in pending_retreats.items():
        province_id = _province_id(province)
        for area in _areas_for_unit(province_id, unit.unit_type):
            board[area, _DISLODGED_NONE_FLAG_INDEX] = 0.0
            if unit.unit_type == UnitType.ARMY:
                board[area, dm_utils.OBSERVATION_DISLODGED_ARMY] = 1.0
            else:
                board[area, dm_utils.OBSERVATION_DISLODGED_FLEET] = 1.0
            board[area, _DISLODGED_NO_OWNER_INDEX] = 0.0
            board[area, _DISLODGED_NO_OWNER_INDEX + _power_index(unit.power)] = 1.0


def _populate_supply_centers(board: np.ndarray, game_state: GameState) -> None:
    for province_name, province in game_state.board.items():
        if not province.is_supply_center:
            continue
        province_id = _province_id(province_name)
        owner = game_state.supply_center_control.get(province_name)
        owner_index = _SC_NO_OWNER_INDEX if owner is None else _SC_INDEX(owner)

        main_area, num_areas = dm_utils.obs_index_start_and_num_areas(province_id)
        board[main_area, _SC_NO_OWNER_INDEX] = 0.0
        board[main_area, owner_index] = 1.0

        # Copy ownership to coastal areas to match DeepMind conventions.
        for offset in range(1, num_areas):
            area = main_area + offset
            board[area, _SC_NO_OWNER_INDEX] = 0.0
            board[area, owner_index] = 1.0


def _build_numbers(game_state: GameState) -> np.ndarray:
    numbers = np.zeros(dm_utils.NUM_POWERS, dtype=np.int32)
    for power, count in game_state.pending_builds.items():
        numbers[_power_index(power)] = np.int32(count)
    for power, count in game_state.pending_disbands.items():
        numbers[_power_index(power)] = np.int32(-count)
    return numbers


def _set_unit_presence(
    board: np.ndarray,
    area: int,
    unit_type: UnitType,
    power: Power,
) -> None:
    board[area, dm_utils.OBSERVATION_UNIT_ABSENT] = 0.0
    if unit_type == UnitType.ARMY:
        board[area, dm_utils.OBSERVATION_UNIT_ARMY] = 1.0
    else:
        board[area, dm_utils.OBSERVATION_UNIT_FLEET] = 1.0
    board[area, _NO_UNIT_OWNER_INDEX] = 0.0
    board[area, dm_utils.OBSERVATION_UNIT_POWER_START + _power_index(power)] = 1.0


def _areas_for_unit(province_id: int, unit_type: UnitType) -> Sequence[int]:
    start, count = dm_utils.obs_index_start_and_num_areas(province_id)
    if count == 1:
        return (start,)
    if unit_type == UnitType.ARMY:
        return (start,)
    # Fleets on bicoastal provinces occupy one of the coastal areas. Without
    # explicit coast tracking we mirror the unit onto both coasts so that the
    # policy receives consistent masking information.
    return tuple(start + offset for offset in range(1, count))


def _province_id(province_name: str) -> int:
    try:
        return _PROVINCE_NAME_TO_ID[province_name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Province {province_name!r} is not part of the standard map") from exc


def _power_index(power: Power) -> int:
    name = str(power)
    try:
        return _POWER_TO_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"Power {name!r} is not supported by the DeepMind policy") from exc


def _SC_INDEX(power: Power) -> int:
    return dm_utils.OBSERVATION_SC_POWER_START + _power_index(power)
