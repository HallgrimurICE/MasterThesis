from __future__ import annotations

"""
Light-weight Diplomacy engine package exposing the public API that used to live
inside ``game_engine.py``.
"""

from .types import Power, Province, Phase, OrderType, Unit, Order, describe_order
from .orders import hold, move, support_hold, support_move, retreat
from .state import GameState
from .adjudication import Resolution, Adjudicator
from .simulation import run_rounds_with_agents
from .maps import build_graph, standard_board, triangle_board, triangle_initial_state
from .agents import (
    Agent,
    ScriptedAgent,
    RandomAgent,
    Directive,
    ObservationBestResponseAgent,
    SampledBestResponsePolicy,
    SaveBestResponseAgent,
)
from .demo import (
    run_standard_board_with_deepmind_turkey,
    run_triangle_board_with_random_agents,
)
from .viz.mesh import (
    visualize_state,
    visualize_state_mesh,
    interactive_visualize_state_mesh,
)

__all__ = [
    "Power",
    "Province",
    "Phase",
    "OrderType",
    "Unit",
    "Order",
    "describe_order",
    "hold",
    "move",
    "support_hold",
    "support_move",
    "retreat",
    "GameState",
    "Resolution",
    "Adjudicator",
    "run_rounds_with_agents",
    "build_graph",
    "standard_board",
    "triangle_board",
    "triangle_initial_state",
    "demo_state_mesh",
    "Agent",
    "ScriptedAgent",
    "RandomAgent",
    "Directive",
    "ObservationBestResponseAgent",
    "SampledBestResponsePolicy",
    "SaveBestResponseAgent",
    "simulate_two_power_cooperation",
    "print_two_power_cooperation_report",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
    "run_standard_board_with_deepmind_turkey",
    "run_triangle_board_with_random_agents",
    "visualize_state",
    "visualize_state_mesh",
    "interactive_visualize_state_mesh",
]
