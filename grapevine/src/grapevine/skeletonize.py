"""PRD §6.2 — Skeletonization.

    Input: Clean masks (merged into single binary vine mask)
    skimage.morphology.skeletonize (Zhang-Suen)
    Output: 1-pixel skeleton

Known issue (documented in PRD): Zhang-Suen produces spurs (short branches)
at junctions — pruned in graph_build.py per §6.3 step 2.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize as _sk_skeletonize


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen skeletonization -> 1px-wide boolean skeleton."""
    return _sk_skeletonize(binary_mask.astype(bool), method="zhang")
