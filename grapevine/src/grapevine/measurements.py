"""PRD §7 — Measurements. All pixel-only (no real-world unit conversion in V1).

| Measurement       | Definition                                                              |
|--------------------|--------------------------------------------------------------------------|
| Trunk length       | Sum of trunk-class edge lengths, root -> first cordon junction           |
| Cordon length       | Sum of cordon-class edge lengths                                         |
| Cane count          | Number of cane-class edges connected to cordon                           |
| Shoot count          | Number of shoot-class terminal edges                                     |
| Branching angle      | Angle between parent/child edge at junction, from skeleton pixel coords  |
"""
from __future__ import annotations

import math

import networkx as nx


def trunk_length_px(g: nx.Graph, root: int) -> float:
    """Sum of trunk-class edge lengths in the vine graph."""
    total = sum(data.get("length_px", 0.0) for _, _, data in g.edges(data=True) if data.get("cls") == "trunk")
    return total


def cordon_length_px(g: nx.Graph) -> float:
    return sum(data.get("length_px", 0.0) for _, _, data in g.edges(data=True) if data.get("cls") == "cordon")


def cane_length_px(g: nx.Graph) -> float:
    return sum(data.get("length_px", 0.0) for _, _, data in g.edges(data=True) if data.get("cls") == "cane")


def shoot_length_px(g: nx.Graph) -> float:
    return sum(data.get("length_px", 0.0) for _, _, data in g.edges(data=True) if data.get("cls") == "shoot")


def trunk_count(g: nx.Graph) -> int:
    """Number of trunk-class branch edges."""
    return sum(1 for _, _, data in g.edges(data=True) if data.get("cls") == "trunk")


def cordon_count(g: nx.Graph) -> int:
    """Number of cordon-class branch edges."""
    return sum(1 for _, _, data in g.edges(data=True) if data.get("cls") == "cordon")


def cane_count(g: nx.Graph) -> int:
    """Number of cane-class branch edges (connected to trunk or cordon)."""
    return sum(1 for _, _, data in g.edges(data=True) if data.get("cls") == "cane")


def shoot_count(g: nx.Graph) -> int:
    """Number of shoot-class branch edges."""
    return sum(1 for _, _, data in g.edges(data=True) if data.get("cls") == "shoot")


def _edge_direction_at_node(g: nx.Graph, junction: int, other: int) -> tuple[float, float]:
    """Direction vector of the edge (junction -> other), using the first
    couple of pixels of pixel_path nearest `junction` for a locally accurate
    angle (skeleton paths can curve)."""
    path = g[junction][other].get("pixel_path")
    jx, jy = g.nodes[junction]["x"], g.nodes[junction]["y"]
    if not path:
        ox, oy = g.nodes[other]["x"], g.nodes[other]["y"]
        return (ox - jx, oy - jy)

    # orient path so it starts at junction
    if path[0] != (jy, jx):
        path = list(reversed(path))
    sample_idx = min(5, len(path) - 1)
    py, px = path[sample_idx]
    return (px - jx, py - jy)


def branching_angles(g: nx.Graph) -> list[dict]:
    """For every junction node, compute angles (degrees) between each pair
    of incident edges, per §7."""
    results = []
    for n, attrs in g.nodes(data=True):
        if attrs.get("type") != "junction":
            continue
        neighbors = list(g.neighbors(n))
        if len(neighbors) < 2:
            continue
        vectors = [_edge_direction_at_node(g, n, nb) for nb in neighbors]
        angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1, v2 = vectors[i], vectors[j]
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                mag1 = math.hypot(*v1)
                mag2 = math.hypot(*v2)
                if mag1 == 0 or mag2 == 0:
                    continue
                cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                angles.append(round(math.degrees(math.acos(cos_a)), 1))
        if angles:
            results.append({"junction": n, "angles": angles})
    return results


def compute_measurements(g: nx.Graph, root: int) -> dict:
    t_len = round(trunk_length_px(g, root), 2)
    c_len = round(cordon_length_px(g), 2)
    cane_len = round(cane_length_px(g), 2)
    s_len = round(shoot_length_px(g), 2)
    return {
        "trunk_length_px": t_len,
        "cordon_length_px": c_len,
        "cane_length_px": cane_len,
        "shoot_length_px": s_len,
        "total_length_px": round(t_len + c_len + cane_len + s_len, 2),
        "trunk_count": trunk_count(g),
        "cordon_count": cordon_count(g),
        "cane_count": cane_count(g),
        "shoot_count": shoot_count(g),
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "branching_angles": branching_angles(g),
    }
