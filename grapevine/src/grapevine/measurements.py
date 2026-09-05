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
    """Sum of trunk-class edges from root to the first cordon junction.

    Walk from root along trunk-class edges only, stopping once a non-trunk
    edge is reached (that's the cordon split point).
    """
    total = 0.0
    current = root
    visited = {root}
    while True:
        trunk_neighbors = [
            (n, data) for n, data in
            ((nb, g[current][nb]) for nb in g.neighbors(current))
            if data.get("cls") == "trunk" and n not in visited
        ]
        if not trunk_neighbors:
            break
        n, data = trunk_neighbors[0]
        total += data["length_px"]
        visited.add(n)
        current = n
    return total


def cordon_length_px(g: nx.Graph) -> float:
    return sum(data["length_px"] for _, _, data in g.edges(data=True) if data.get("cls") == "cordon")


def cane_count(g: nx.Graph) -> int:
    """Number of cane-class edges directly connected to a cordon-class edge
    (i.e. cane edges originating at a node that also touches a cordon edge)."""
    count = 0
    for u, v, data in g.edges(data=True):
        if data.get("cls") != "cane":
            continue
        touches_cordon = any(
            g[n][nb].get("cls") == "cordon"
            for n in (u, v)
            for nb in g.neighbors(n)
            if nb not in (u, v)
        )
        if touches_cordon:
            count += 1
    return count


def shoot_count(g: nx.Graph) -> int:
    """Number of shoot-class terminal edges (edge touches a degree-1 node)."""
    count = 0
    for u, v, data in g.edges(data=True):
        if data.get("cls") != "shoot":
            continue
        if g.degree(u) == 1 or g.degree(v) == 1:
            count += 1
    return count


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
    return {
        "trunk_length_px": round(trunk_length_px(g, root), 2),
        "cordon_length_px": round(cordon_length_px(g), 2),
        "cane_count": cane_count(g),
        "shoot_count": shoot_count(g),
        "branching_angles": branching_angles(g),
    }
