"""PRD §6.3 — Graph Construction.

    Input: Skeleton
    1. Detect nodes: 3x3 neighborhood pixel count
       - Endpoint: 1 neighbor
       - Junction: >=3 neighbors
       - Regular: 2 neighbors (part of edge)
    2. Prune spurs: Remove terminal branches <10px unless they're shoot tips
    3. Build NetworkX graph: nodes = endpoints/junctions, edges = pixel paths
    4. Label edges by dominant mask class along path
    Output: NetworkX Graph with 'class' edge attribute

Implementation note: a real skeleton intersection is rarely a single pixel —
Zhang-Suen thinning typically produces a small *cluster* of adjacent pixels
that each satisfy the ">=3 neighbors" junction test at a single T/Y
intersection. Treating each of those pixels as its own node would sever the
branches meeting there into a knot of ~1px "edges" and disconnect the real
long branches on either side. So junction pixels that are 8-connected to each
other are merged into one node (§6.3 step 1 still applies per-pixel; the
merge is what makes step 3's "nodes = junctions" produce one node per real
intersection instead of one per pixel).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage

CLASS_NAMES = {0: "background", 1: "trunk", 2: "cordon", 3: "cane", 4: "shoot"}
MIN_SPUR_LENGTH = 10  # px, per §6.3 step 2

_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbor_count(skel: np.ndarray, y: int, x: int) -> int:
    h, w = skel.shape
    count = 0
    for dy, dx in _NEIGHBORS_8:
        ny, nx_ = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx_ < w and skel[ny, nx_]:
            count += 1
    return count


def detect_node_pixels(skel: np.ndarray) -> tuple[set, set]:
    """Step 1. Returns (endpoint_pixels, junction_pixels) as sets of (y, x)."""
    ys, xs = np.nonzero(skel)
    endpoints, junctions = set(), set()
    for y, x in zip(ys, xs):
        n = _neighbor_count(skel, y, x)
        if n == 1:
            endpoints.add((y, x))
        elif n >= 3:
            junctions.add((y, x))
    return endpoints, junctions


def _cluster_junction_pixels(junctions: set, shape: tuple[int, int]) -> dict:
    """8-connect junction pixels into clusters. Returns {pixel: cluster_label}."""
    if not junctions:
        return {}
    mask = np.zeros(shape, dtype=bool)
    for y, x in junctions:
        mask[y, x] = True
    structure = np.ones((3, 3), dtype=int)
    labeled, _n = ndimage.label(mask, structure=structure)
    return {(y, x): int(labeled[y, x]) for y, x in junctions}


def _build_node_index(endpoints: set, junctions: set, shape: tuple[int, int]):
    """Assigns a stable integer node id to every endpoint (1 pixel = 1 node)
    and every junction cluster (N pixels = 1 node). Returns:
        pixel_to_node: {(y,x): node_id}
        node_members:  {node_id: set of (y,x)}
        node_kind:     {node_id: "endpoint" | "junction"}
    """
    pixel_to_node: dict[tuple, int] = {}
    node_members: dict[int, set] = {}
    node_kind: dict[int, str] = {}
    next_id = 0

    for p in endpoints:
        pixel_to_node[p] = next_id
        node_members[next_id] = {p}
        node_kind[next_id] = "endpoint"
        next_id += 1

    cluster_map = _cluster_junction_pixels(junctions, shape)
    label_to_node: dict[int, int] = {}
    for p, label in cluster_map.items():
        if label not in label_to_node:
            label_to_node[label] = next_id
            node_members[next_id] = set()
            node_kind[next_id] = "junction"
            next_id += 1
        nid = label_to_node[label]
        pixel_to_node[p] = nid
        node_members[nid].add(p)

    return pixel_to_node, node_members, node_kind


def _node_centroid(members: set) -> tuple[int, int]:
    ys = [p[0] for p in members]
    xs = [p[1] for p in members]
    return (int(round(sum(ys) / len(ys))), int(round(sum(xs) / len(xs))))


def _path_length(path: list[tuple]) -> float:
    length = 0.0
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        length += np.hypot(y1 - y0, x1 - x0)
    return length


def _dominant_class(path: list[tuple], class_map: np.ndarray) -> str:
    votes = np.zeros(5, dtype=int)  # 0..4
    for y, x in path:
        votes[class_map[y, x]] += 1
    votes[0] = 0  # ignore background votes when picking dominant structure class
    idx = int(np.argmax(votes)) if votes.sum() > 0 else 0
    return CLASS_NAMES[idx]


def _walk_corridor(skel: np.ndarray, start: tuple, first_step: tuple, pixel_to_node: dict):
    """Walk a chain of 'regular' (degree-2) skeleton pixels starting at
    `start`, first hop `first_step`, until reaching any node pixel. Returns
    the full pixel path including both ends."""
    h, w = skel.shape
    path = [start, first_step]
    prev, cur = start, first_step
    while cur not in pixel_to_node:
        candidates = []
        for dy, dx in _NEIGHBORS_8:
            ny, nx_ = cur[0] + dy, cur[1] + dx
            if not (0 <= ny < h and 0 <= nx_ < w):
                continue
            if not skel[ny, nx_]:
                continue
            if (ny, nx_) == prev:
                continue
            candidates.append((ny, nx_))
        if not candidates:
            break  # dead end without reaching a node (shouldn't normally happen)
        nxt = candidates[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def build_edge_paths(skel, pixel_to_node, node_members):
    """Steps 1-3 groundwork: enumerate raw pixel paths between distinct nodes,
    walking through corridors of regular pixels and skipping internal
    same-cluster adjacency."""
    h, w = skel.shape
    visited_steps: set[tuple] = set()  # undirected (p, q) pairs already consumed
    edges = []  # list of (node_a, node_b, pixel_path)

    for node_id, members in node_members.items():
        for p in members:
            for dy, dx in _NEIGHBORS_8:
                q = (p[0] + dy, p[1] + dx)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or not skel[q]:
                    continue
                if (p, q) in visited_steps:
                    continue
                if q in pixel_to_node:
                    other = pixel_to_node[q]
                    if other == node_id:
                        visited_steps.add((p, q))
                        visited_steps.add((q, p))
                        continue  # internal adjacency within same cluster
                    visited_steps.add((p, q))
                    visited_steps.add((q, p))
                    edges.append((node_id, other, [p, q]))
                    continue
                # q is a regular corridor pixel -> walk it out to the next node
                path = _walk_corridor(skel, p, q, pixel_to_node)
                for a, b in zip(path[:-1], path[1:]):
                    visited_steps.add((a, b))
                    visited_steps.add((b, a))
                end = path[-1]
                if end in pixel_to_node:
                    other = pixel_to_node[end]
                    if other != node_id:
                        edges.append((node_id, other, path))
    return edges


def prune_spurs(edges: list[tuple], node_kind: dict, class_map: np.ndarray) -> list[tuple]:
    """Step 2. Remove terminal branches < MIN_SPUR_LENGTH px, unless the
    branch's dominant class is 'shoot' (shoot tips are legitimate structure,
    per PRD: 'unless they're shoot tips')."""
    kept = []
    for a, b, path in edges:
        is_terminal = node_kind.get(a) == "endpoint" or node_kind.get(b) == "endpoint"
        if not is_terminal or _path_length(path) >= MIN_SPUR_LENGTH:
            kept.append((a, b, path))
            continue
        if _dominant_class(path, class_map) == "shoot":
            kept.append((a, b, path))  # keep short shoot tips
        # else: drop as spur artifact
    return kept


def build_graph(skel: np.ndarray, class_map: np.ndarray) -> nx.Graph:
    """Full §6.3 pipeline. class_map: H×W uint8, values per
    postprocess.class_label_map (0=bg,1=trunk,2=cordon,3=cane,4=shoot)."""
    endpoints, junctions = detect_node_pixels(skel)
    pixel_to_node, node_members, node_kind = _build_node_index(endpoints, junctions, skel.shape)

    raw_edges = build_edge_paths(skel, pixel_to_node, node_members)
    edges = prune_spurs(raw_edges, node_kind, class_map)

    g = nx.Graph()
    for node_id, members in node_members.items():
        y, x = _node_centroid(members)
        g.add_node(node_id, x=x, y=y, type=node_kind[node_id])

    for a, b, path in edges:
        cls = _dominant_class(path, class_map)
        length = _path_length(path)
        if g.has_edge(a, b) and length <= g[a][b]["length_px"]:
            continue
        g.add_edge(a, b, cls=cls, length_px=length, pixel_path=path)

    # Drop nodes that ended up with no surviving edges (pruned spurs can
    # orphan a node entirely) so downstream code doesn't trip on isolates
    # that aren't real structure.
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    g.remove_nodes_from(isolated)

    return g
