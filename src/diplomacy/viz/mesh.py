from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import colors as mcolors  # type: ignore
    from matplotlib.widgets import Button, Slider  # type: ignore

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback used when matplotlib missing
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    mcolors = None  # type: ignore
    Button = None  # type: ignore
    Slider = None  # type: ignore

from ..graph import NX_AVAILABLE, nx
from ..maps import build_graph
from ..state import GameState
from ..types import Power, ProvinceType


POWER_COLORS: Dict[str, str] = {
    "Blue": "#1f77b4",
    "Pink": "#ff69b4",
    "Red": "#d62728",
    "Green": "#2ca02c",
    "Yellow": "#ffbf00",
    "Austria": "#d62728",
    "Turkey": "#ffbf00",
    "England": "#2f6bdc",
    "Russia": "#ff69b4",
    "Italy": "#27ae60",
    "France": "#3498db",
}


def _power_color(power: Power) -> str:
    return POWER_COLORS.get(str(power), "#666666")


def _power_text_color(color: str) -> str:
    if mcolors is None:  # pragma: no cover - fallback when matplotlib missing
        return "white"
    r, g, b = mcolors.to_rgb(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"


def _ensure_graph(state: GameState) -> None:
    if len(state.graph.nodes()) == len(state.board):
        return
    state.graph = build_graph(state.board)


def visualize_state(state: GameState, title: str = "Board State"):
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError("networkx and matplotlib are required for visualization utilities.")
    _ensure_graph(state)
    pos = nx.spring_layout(state.graph, seed=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx(state.graph, pos, ax=ax, with_labels=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _mesh_positions() -> Dict[str, Tuple[float, float]]:
    grid_pos = {
        "1": (0, 2),
        "2": (1, 2),
        "3": (2, 2),
        "4": (3, 2),
        "5": (4, 2),
        "6": (0, 1),
        "7": (1, 1),
        "8": (2, 1),
        "9": (3, 1),
        "10": (4, 1),
        "11": (0, 0),
        "12": (1, 0),
        "13": (2, 0),
        "14": (3, 0),
        "15": (4, 0),
    }
    return {k: (v[0] * 1.2, v[1] * 1.0) for k, v in grid_pos.items()}


def _positions_for_state(state: GameState) -> Dict[str, Tuple[float, float]]:
    mesh_pos = _mesh_positions()
    nodes = list(state.graph.nodes())
    if nodes and all(n in mesh_pos for n in nodes):
        return {n: mesh_pos[n] for n in nodes}
    if not NX_AVAILABLE:
        raise RuntimeError("networkx is required for automatic layout of non-mesh maps.")
    layout = nx.spring_layout(state.graph, seed=42)
    return {node: (float(coord[0]), float(coord[1])) for node, coord in layout.items()}


def _draw_mesh_state(ax, state: GameState, pos: Dict[str, Tuple[float, float]], title: str) -> None:
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError("networkx and matplotlib are required for visualization utilities.")
    _ensure_graph(state)
    ax.clear()
    nx.draw_networkx_edges(state.graph, pos, ax=ax)

    sc_nodes = [n for n, d in state.graph.nodes(data=True) if d.get("is_supply_center", False)]
    non_sc = [n for n in state.graph.nodes() if n not in sc_nodes]
    sea_nodes = [
        n for n, d in state.graph.nodes(data=True) if d.get("province_type") == ProvinceType.SEA
    ]
    nx.draw_networkx_nodes(
        state.graph,
        pos,
        nodelist=[n for n in non_sc if n not in sea_nodes],
        node_color="white",
        edgecolors="black",
        linewidths=1,
        node_size=1200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        state.graph,
        pos,
        nodelist=sea_nodes,
        node_color="#a6cee3",
        edgecolors="black",
        linewidths=1,
        node_size=1200,
        ax=ax,
    )

    sc_colors: List[str] = []
    sc_edge_colors: List[str] = []
    for node in sc_nodes:
        controller = state.supply_center_control.get(node)
        if controller is not None:
            sc_colors.append(_power_color(controller))
            sc_edge_colors.append("black")
        else:
            sc_colors.append("#f5f5f5")
            sc_edge_colors.append("black")

    nx.draw_networkx_nodes(
        state.graph,
        pos,
        nodelist=sc_nodes,
        node_color=sc_colors,
        edgecolors=sc_edge_colors,
        linewidths=1,
        node_size=1200,
        ax=ax,
    )

    # draw supply centers as inner markers
    for node in sc_nodes:
        x, y = pos[node]
        ax.scatter(
            [x],
            [y],
            s=170,
            c="#ffffff",
            zorder=9,
            edgecolors="red",
            linewidths=1.2,
            marker="o",
        )
    label_artists = nx.draw_networkx_labels(state.graph, pos, font_size=9, ax=ax)
    for text in label_artists.values():
        text.set_zorder(12)

    for loc, unit in state.units.items():
        x, y = pos[loc]
        color = _power_color(unit.power)
        text_color = _power_text_color(color)
        marker = "^" if unit.unit_type.name == "FLEET" else "o"
        ax.scatter([x], [y], s=350, c=color, zorder=10, edgecolors="black", linewidths=1, marker=marker)
        ax.text(x, y, f"{unit.power[0]}", ha="center", va="center", fontsize=10, color=text_color, zorder=11)

    ax.set_title(title)
    ax.axis("off")


def visualize_state_mesh(state: GameState, title: str = "5x3 Mesh Map"):
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None:
        raise RuntimeError("networkx and matplotlib are required for visualization utilities.")
    _ensure_graph(state)
    pos = _positions_for_state(state)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    _draw_mesh_state(ax, state, pos, title)
    plt.tight_layout()
    plt.show()


def interactive_visualize_state_mesh(states: List[GameState], titles: Optional[List[str]] = None) -> None:
    if not states:
        return
    if titles is None or len(titles) != len(states):
        titles = [f"Round {i}" for i in range(len(states))]
    if not NX_AVAILABLE or not MATPLOTLIB_AVAILABLE or plt is None or Button is None:
        raise RuntimeError("networkx and matplotlib are required for visualization utilities.")

    for state in states:
        _ensure_graph(state)

    pos = _positions_for_state(states[0])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    plt.subplots_adjust(bottom=0.25)

    state_index = {"value": 0}
    button_styles: Dict[str, Tuple[str, str]] = {}

    axprev = plt.axes([0.3, 0.05, 0.15, 0.08])
    axnext = plt.axes([0.55, 0.05, 0.15, 0.08])
    bprev = Button(axprev, "Previous")
    bnext = Button(axnext, "Next")
    button_styles["prev"] = (bprev.color, bprev.hovercolor)
    button_styles["next"] = (bnext.color, bnext.hovercolor)

    slider_ax = plt.axes([0.15, 0.01, 0.7, 0.03])
    slider = Slider(
        slider_ax,
        "Round",
        0,
        len(states) - 1,
        valinit=0,
        valstep=1,
        color="#cccccc",
    )
    slider_ax.xaxis.set_ticks_position("bottom")
    slider_ax.yaxis.set_visible(False)
    slider_ax.set_facecolor("#f2f2f2")

    slider_updating = {"active": False}

    def _set_button_enabled(button: Button, *, enabled: bool, base_color: str, base_hover: str) -> None:
        button.eventson = enabled
        face_color = base_color if enabled else "#d3d3d3"
        hover_color = base_hover if enabled else "#c0c0c0"
        button.ax.patch.set_facecolor(face_color)
        button.hovercolor = hover_color
        label_color = "black" if enabled else "#6f6f6f"
        button.label.set_color(label_color)

    def update_buttons() -> None:
        at_start = state_index["value"] == 0
        at_end = state_index["value"] == len(states) - 1
        base_prev, base_prev_hover = button_styles["prev"]
        base_next, base_next_hover = button_styles["next"]
        _set_button_enabled(bprev, enabled=not at_start, base_color=base_prev, base_hover=base_prev_hover)
        _set_button_enabled(bnext, enabled=not at_end, base_color=base_next, base_hover=base_next_hover)

    def draw_current(*, update_slider: bool = True) -> None:
        idx = state_index["value"]
        _draw_mesh_state(ax, states[idx], pos, titles[idx])
        update_buttons()
        if update_slider and not slider_updating["active"]:
            slider_updating["active"] = True
            slider.set_val(idx)
            slider_updating["active"] = False
        fig.canvas.draw_idle()

    def go_next(_event) -> None:
        if state_index["value"] < len(states) - 1:
            state_index["value"] += 1
            draw_current()

    def go_prev(_event) -> None:
        if state_index["value"] > 0:
            state_index["value"] -= 1
            draw_current()

    bprev.on_clicked(go_prev)
    bnext.on_clicked(go_next)

    def slider_changed(val: float) -> None:
        if slider_updating["active"]:
            return
        idx = int(round(val))
        idx = max(0, min(len(states) - 1, idx))
        if idx == state_index["value"]:
            return
        state_index["value"] = idx
        draw_current(update_slider=False)

    slider.on_changed(slider_changed)

    draw_current()
    plt.show()


__all__ = ["visualize_state", "visualize_state_mesh", "interactive_visualize_state_mesh"]
