import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))



from diplomacy.maps import standard_initial_state
from diplomacy.viz.mesh import interactive_visualize_state_mesh
from diplomacy.types import Unit, UnitType, Power

state = standard_initial_state()
state.units["TUN"] = Unit(Power("Italy"), "TUN", UnitType.FLEET)
state.units["MAR"] = Unit(Power("Italy"), "MAR", UnitType.ARMY)
state.supply_center_control["TUN"] = Power("Italy")
state.supply_center_control["MAR"] = Power("Italy")

interactive_visualize_state_mesh([state], ["Italy controls MAR+TUN with extra units"])
PY
