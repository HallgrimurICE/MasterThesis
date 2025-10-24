"""
Backwards-compatible facade re-exporting the Diplomacy engine components that
now live in the ``diplomacy`` package.
"""

from diplomacy import *  # noqa: F401,F403
from diplomacy import __all__ as _diplomacy_all
from diplomacy.demo import (
    demo_run_mesh_with_random_agents,
    demo_run_mesh_with_random_orders,
    print_two_power_cooperation_report,
    simulate_two_power_cooperation,
)
from diplomacy.viz.mesh import (
    interactive_visualize_state_mesh,
    visualize_state,
    visualize_state_mesh,
)

__all__ = list(_diplomacy_all) + [
    "visualize_state",
    "visualize_state_mesh",
    "interactive_visualize_state_mesh",
    "simulate_two_power_cooperation",
    "print_two_power_cooperation_report",
    "demo_run_mesh_with_random_orders",
    "demo_run_mesh_with_random_agents",
]
