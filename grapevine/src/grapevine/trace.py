"""PRD §6.4 — Whole-Vine Tracing.

    Input: Graph
    1. Find root: Lowest-Y junction with trunk-class edge (image coords, Y increases downward)
       -> i.e. the *largest* y value, since root sits at ground level, bottom of frame.
    2. BFS from root
    3. Enforce hierarchy: trunk -> cordon -> cane -> shoot
       If BFS encounters cane before cordon, flag as error (F4 or F3)
    4. Output: Ordered list of paths from root to each leaf

Multiple vines: if the graph has >1 disconnected component with trunk-class
edges, process each separately. Output array of vine objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

HIERARCHY = {"trunk": 0, "cordon": 1, "cane": 2, "shoot": 3}


@dataclass
class TraceResult:
    root: int
    paths: list[dict]          # [{"nodes": [...], "classes": [...], "total_length_px": float}]
    hierarchy_violations: list[dict]  # [{"edge": (u, v), "code": "F3"|"F4", "detail": str}]


def find_root(g: nx.Graph) -> int | None:
    """Step 1: lowest point of the trunk (max pixel y, i.e. closest to
    ground/bottom of frame) among nodes touching a trunk-class edge.

    Prefer a junction if one sits at the bottom of the trunk (e.g. a graft
    union or an in-frame branch point); otherwise the trunk simply ends at
    an endpoint at ground level / the image edge, which is the natural root.
    Per PRD §6.4: "Lowest-Y junction with trunk-class edge" — we widen this
    to "lowest-Y node" since a real trunk base is usually a dead end
    (endpoint), not a junction, unless the frame happens to end exactly at
    the cordon split.
    """
    candidates = []
    for u, v, data in g.edges(data=True):
        if data.get("cls") != "trunk":
            continue
        candidates.extend([u, v])
    if not candidates:
        return None
    return max(candidates, key=lambda n: g.nodes[n]["y"])


def _leaf_nodes(g: nx.Graph, root: int) -> list[int]:
    return [n for n in g.nodes if g.degree(n) == 1 and n != root]


def trace_vine(g: nx.Graph) -> TraceResult | None:
    """Steps 2-4 for a single connected component graph `g`."""
    root = find_root(g)
    if root is None:
        return None

    violations = []
    # BFS from root, tracking the max hierarchy level seen so far on each path
    # to catch out-of-order transitions (e.g. cane appearing before cordon).
    predecessors = {root: None}
    max_level_seen = {root: -1}  # trunk starts at level 0, so -1 means "nothing yet"
    order = list(nx.bfs_edges(g, root))

    for u, v in order:
        predecessors[v] = u
        edge_cls = g[u][v]["cls"]
        level = HIERARCHY.get(edge_cls, 0)
        prev_max = max_level_seen[u]
        if level < prev_max:
            # Encountered a lower-hierarchy class after a higher one on this
            # path, e.g. cane (2) appearing after we'd already reached shoot (3)
            # level, or cane appearing before any cordon (skip from trunk->cane).
            code = "F4" if level == 0 else "F3"
            violations.append({
                "edge": (u, v),
                "code": code,
                "detail": f"'{edge_cls}' (level {level}) followed level {prev_max} "
                          f"on the path from root; hierarchy should be non-decreasing.",
            })
        # skipped level (e.g. trunk directly to cane, skipping cordon) is also
        # a hierarchy anomaly worth flagging even though not strictly "before"
        elif level > prev_max + 1:
            violations.append({
                "edge": (u, v),
                "code": "F3",
                "detail": f"'{edge_cls}' (level {level}) skips a hierarchy level "
                          f"(previous max level {prev_max}); possible mislabeled class.",
            })
        max_level_seen[v] = max(prev_max, level)

    leaves = _leaf_nodes(g, root)
    paths_out = []
    for leaf in leaves:
        try:
            node_path = nx.shortest_path(g, root, leaf)
        except nx.NetworkXNoPath:
            continue
        classes = [g[node_path[i]][node_path[i + 1]]["cls"] for i in range(len(node_path) - 1)]
        total_len = sum(g[node_path[i]][node_path[i + 1]]["length_px"] for i in range(len(node_path) - 1))
        paths_out.append({"nodes": node_path, "classes": classes, "total_length_px": total_len})

    return TraceResult(root=root, paths=paths_out, hierarchy_violations=violations)


def split_vines(g: nx.Graph) -> list[nx.Graph]:
    """Multiple vines: split into connected components, keep only components
    that contain at least one trunk-class edge."""
    components = []
    for comp_nodes in nx.connected_components(g):
        sub = g.subgraph(comp_nodes).copy()
        if any(data.get("cls") == "trunk" for _, _, data in sub.edges(data=True)):
            components.append(sub)
    return components


def trace_all_vines(g: nx.Graph) -> list[TraceResult]:
    results = []
    for sub in split_vines(g):
        r = trace_vine(sub)
        if r is not None:
            results.append(r)
    return results
