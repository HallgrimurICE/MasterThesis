from __future__ import annotations

import os
import sys
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from diplomacy.demo import play_support_br_against_random_agents
from diplomacy.state import GameState
from diplomacy.types import Power, Province, Unit

from best_response_agent import propose_bundles, sbr_with_supports


class FakeAdjudicator:
    def __init__(self, legal_map: Dict[str, list[str]]) -> None:
        self.legal_map = legal_map
        self.apply_calls = 0

    def legal_orders(self, state: GameState, unit: Unit) -> list[str]:
        return list(self.legal_map.get(unit.loc, []))

    def apply_orders(self, state: GameState, joint_orders):
        self.apply_calls += 1
        return state


class FakePolicy:
    def __init__(self, adjudicator: FakeAdjudicator, opponent_orders: Dict[Unit, str] | None = None) -> None:
        self.adjudicator = adjudicator
        self.opponent_orders = opponent_orders or {}
        self.sample_calls: list[float] = []

    def sample_joint_orders(self, state: GameState, *, exclude_power=None, temperature: float = 0.7):
        self.sample_calls.append(temperature)
        return dict(self.opponent_orders)


def build_state(province_defs: Dict[str, tuple[list[str], bool]], units: Dict[str, Unit]) -> GameState:
    board = {
        name: Province(name=name, neighbors=set(neighbors), is_supply_center=is_sc)
        for name, (neighbors, is_sc) in province_defs.items()
    }
    powers = {unit.power for unit in units.values()}
    return GameState(board=board, units=dict(units), powers=powers)


def test_support_hold_bundle_created():
    france = Power("FRANCE")
    province_defs = {
        "PAR": (["MAR"], True),
        "MAR": (["PAR"], False),
    }
    units = {
        "PAR": Unit(france, "PAR"),
        "MAR": Unit(france, "MAR"),
    }
    state = build_state(province_defs, units)
    adjudicator = FakeAdjudicator(
        {
            "PAR": ["A PAR H"],
            "MAR": ["A MAR H", "A MAR S PAR H"],
        }
    )
    policy = FakePolicy(adjudicator)
    bundles = propose_bundles(state, france, policy)

    par_unit = state.units["PAR"]
    mar_unit = state.units["MAR"]

    assert any(
        bundle.get(par_unit) == "A PAR H" and bundle.get(mar_unit) == "A MAR S PAR H"
        for bundle in bundles
    )


def test_support_move_bundle_created():
    france = Power("FRANCE")
    province_defs = {
        "PAR": (["MAR", "BUR"], True),
        "MAR": (["PAR", "BUR"], False),
        "BUR": (["PAR", "MAR"], True),
    }
    units = {
        "PAR": Unit(france, "PAR"),
        "MAR": Unit(france, "MAR"),
        "BUR": Unit(Power("GERMANY"), "BUR"),
    }
    state = build_state(province_defs, units)
    adjudicator = FakeAdjudicator(
        {
            "PAR": ["A PAR H", "A PAR - BUR"],
            "MAR": ["A MAR H", "A MAR S PAR - BUR"],
        }
    )
    policy = FakePolicy(adjudicator)
    bundles = propose_bundles(state, france, policy)

    par_unit = state.units["PAR"]
    mar_unit = state.units["MAR"]

    assert any(
        bundle.get(par_unit) == "A PAR - BUR" and bundle.get(mar_unit) == "A MAR S PAR - BUR"
        for bundle in bundles
    )


def test_sbr_with_supports_uses_apply_and_value_fn():
    france = Power("FRANCE")
    province_defs = {"PAR": ([], True)}
    units = {"PAR": Unit(france, "PAR")}
    state = build_state(province_defs, units)
    adjudicator = FakeAdjudicator({"PAR": ["A PAR H"]})
    policy = FakePolicy(adjudicator)

    value_calls: list[float] = []

    def value_fn(next_state: GameState, *, for_power: Power) -> float:
        value_calls.append(1.0)
        return 0.0

    bundle = sbr_with_supports(
        state,
        france,
        policy,
        value_fn,
        B_base=3,
        C_cand=5,
        temperature_grid=(0.5,),
    )

    assert bundle == {state.units["PAR"]: "A PAR H"}
    assert adjudicator.apply_calls == 3
    assert len(value_calls) == 3


def test_support_br_demo_runs(capsys):
    play_support_br_against_random_agents(
        rounds=1,
        seed=0,
        br_kwargs={"B_base": 2, "C_cand": 6, "temperature_grid": (0.5,)},
    )
    captured = capsys.readouterr()
    assert "Support-aware Best Response Demo" in captured.out

