"""PRD §9 — Visualization Specification.

Single output PNG, 2x2 panel layout:

    +-----------+-------------+
    | Original  | Segmentation |
    |  Image    |   Overlay    |
    +-----------+-------------+
    | Skeleton  |   Graph      |
    |           |  + Measure   |
    +-----------+-------------+

Colors/sizes are fixed per the spec table — do not parameterize.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import networkx as nx
import numpy as np

COLORS = {
    "trunk": "#8B4513",   # saddlebrown
    "cordon": "#D2691E",  # chocolate
    "cane": "#CD853F",    # peru
    "shoot": "#32CD32",   # limegreen
}
LINE_WIDTH = {"trunk": 3, "cordon": 2, "cane": 2, "shoot": 1}

JUNCTION_STYLE = dict(radius=4, facecolor="white", edgecolor="black")
ENDPOINT_STYLE = dict(radius=3, facecolor="gray", edgecolor="black")


def _draw_overlay(ax, image: np.ndarray, class_map: np.ndarray):
    ax.imshow(image)
    rgba = np.zeros((*class_map.shape, 4))
    idx_to_cls = {1: "trunk", 2: "cordon", 3: "cane", 4: "shoot"}
    for idx, cls in idx_to_cls.items():
        color = _hex_to_rgb(COLORS[cls])
        mask = class_map == idx
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
        rgba[mask, 3] = 0.6
    ax.imshow(rgba)
    ax.set_title("Segmentation Overlay")
    ax.axis("off")


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _draw_skeleton(ax, skel: np.ndarray):
    ax.imshow(skel, cmap="gray")
    ax.set_title("Skeleton")
    ax.axis("off")


def _draw_graph(ax, image_shape, graph: nx.Graph, measurements_by_vine: dict[int, dict], roots: dict[int, int]):
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # image coords, y down
    ax.set_facecolor("black")

    for u, v, data in graph.edges(data=True):
        cls = data.get("cls", "shoot")
        x0, y0 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x1, y1 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        ax.plot([x0, x1], [y0, y1], color=COLORS.get(cls, "#32CD32"),
                linewidth=LINE_WIDTH.get(cls, 1))

    for n, attrs in graph.nodes(data=True):
        style = JUNCTION_STYLE if attrs.get("type") == "junction" else ENDPOINT_STYLE
        circle = plt.Circle((attrs["x"], attrs["y"]), radius=style["radius"],
                             facecolor=style["facecolor"], edgecolor=style["edgecolor"],
                             linewidth=1, zorder=5)
        ax.add_patch(circle)

    for vine_id, root in roots.items():
        m = measurements_by_vine.get(vine_id, {})
        rx, ry = graph.nodes[root]["x"], graph.nodes[root]["y"]
        text = (f"vine {vine_id}: trunk {m.get('trunk_length_px', 0):.0f}px, "
                f"cordon {m.get('cordon_length_px', 0):.0f}px, "
                f"{m.get('cane_count', 0)} canes, {m.get('shoot_count', 0)} shoots")
        ax.text(rx, max(ry - 15, 10), text, color="white", fontsize=10,
                 path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    ax.set_title("Graph + Measurements")
    ax.invert_yaxis()
    ax.axis("off")


def render_overlay_png(
    image: np.ndarray,
    class_map: np.ndarray,
    skeleton: np.ndarray,
    graph: nx.Graph,
    measurements_by_vine: dict[int, dict],
    roots: dict[int, int],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    _draw_overlay(axes[0, 1], image, class_map)
    _draw_skeleton(axes[1, 0], skeleton)
    _draw_graph(axes[1, 1], image.shape[:2], graph, measurements_by_vine, roots)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
