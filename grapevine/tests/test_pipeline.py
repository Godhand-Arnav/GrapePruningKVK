"""Smoke tests for the non-model half of the pipeline (§6-§7): feed a
synthetic 'T-shaped vine' mask through cleanup -> skeleton -> graph -> trace
-> measurements and check it doesn't blow up and produces sane output.

Run: pytest tests/test_pipeline.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grapevine import measurements as meas
from grapevine import postprocess as pp
from grapevine import skeletonize as skel_mod
from grapevine import trace as trace_mod
from grapevine.graph_build import build_graph


def _synthetic_masks(size=200):
    """Trunk running up the middle, cordon branching left/right at the top,
    one cane + one shoot off the right cordon. Strips are drawn >=8px wide so
    they survive the disk(r=2) morphological opening in §6.1 cleanup — real
    trunk/cordon/cane widths are well above the opening radius; only shoots
    thinner than ~5px are expected to be affected, per PRD §2/§6.1.
    """
    trunk = np.zeros((size, size), dtype=bool)
    trunk[100:190, 96:104] = True  # vertical trunk, bottom half, 8px wide

    cordon = np.zeros((size, size), dtype=bool)
    cordon[96:104, 40:160] = True  # horizontal cordon at the trunk top, 8px wide

    cane = np.zeros((size, size), dtype=bool)
    cane[60:100, 146:154] = True  # cane going up from right cordon end, 8px wide

    shoot = np.zeros((size, size), dtype=bool)
    shoot[20:60, 146:154] = True  # shoot continuing up from cane, 8px wide

    return {"trunk": trunk, "cordon": cordon, "cane": cane, "shoot": shoot}


def test_full_chain_runs_and_finds_root():
    raw = _synthetic_masks()
    clean = pp.clean_masks(raw)
    combined = pp.combined_binary_mask(clean)
    class_map = pp.class_label_map(clean)

    skeleton = skel_mod.skeletonize(combined)
    assert skeleton.sum() > 0, "skeleton should not be empty"

    graph = build_graph(skeleton, class_map)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    traces = trace_mod.trace_all_vines(graph)
    assert len(traces) == 1, "expected exactly one trunk-bearing component"

    tr = traces[0]
    assert tr.root is not None
    # root should be near the bottom of the frame (max y among trunk nodes)
    root_y = graph.nodes[tr.root]["y"]
    assert root_y > 150, f"root should be near image bottom, got y={root_y}"

    m = meas.compute_measurements(graph.subgraph({n for p in tr.paths for n in p["nodes"]} | {tr.root}), tr.root)
    assert m["trunk_length_px"] > 0
    assert m["cordon_length_px"] > 0


def test_class_priority_merge_resolves_overlap():
    size = 50
    trunk = np.zeros((size, size), dtype=bool)
    trunk[10:40, 10:40] = True
    shoot = np.zeros((size, size), dtype=bool)
    shoot[10:40, 10:40] = True  # fully overlapping with trunk

    merged = pp.merge_by_priority({"trunk": trunk, "shoot": shoot})
    assert merged["trunk"].sum() == trunk.sum()
    assert merged["shoot"].sum() == 0, "trunk has priority over shoot on overlapping pixels"
