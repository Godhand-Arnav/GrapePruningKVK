"""PRD §6.1 — Mask Cleanup.

    Input: YOLO-seg masks (4 classes, H×W)
    1. Morphological opening (disk r=2) per class -> remove noise <4px
    2. Remove connected components <20px area per class
    3. Fill holes <50px area per class
    4. Class priority merge: trunk > cordon > cane > shoot
       (if two classes overlap after cleanup, higher priority wins)
    Output: Clean masks

Parameters are fixed per the PRD ("No runtime tuning. If they fail, retrain
or adjust at model level.") — do not expose them as CLI flags.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, binary_opening, remove_small_objects
from skimage.morphology import remove_small_holes

CLASSES = ["trunk", "cordon", "cane", "shoot"]  # priority order, highest first

_OPENING_RADIUS = 2       # disk r=2
_MIN_COMPONENT_AREA = 20  # px
_MIN_HOLE_AREA = 50       # px


def clean_class_mask(mask: np.ndarray) -> np.ndarray:
    """Apply steps 1-3 to a single binary class mask."""
    mask = mask.astype(bool)

    # 1. Morphological opening, disk r=2
    mask = binary_opening(mask, footprint=disk(_OPENING_RADIUS))

    # 2. Remove connected components < 20px area
    try:
        mask = remove_small_objects(mask, max_size=_MIN_COMPONENT_AREA - 1)
    except TypeError:  # older skimage without max_size
        mask = remove_small_objects(mask, min_size=_MIN_COMPONENT_AREA)

    # 3. Fill holes < 50px area
    try:
        mask = remove_small_holes(mask, max_size=_MIN_HOLE_AREA - 1)
    except TypeError:  # older skimage without max_size
        mask = remove_small_holes(mask, area_threshold=_MIN_HOLE_AREA)

    return mask


def merge_by_priority(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Step 4: class priority merge. trunk > cordon > cane > shoot.

    `masks` maps class name -> cleaned binary mask (all same H×W).
    Returns new masks where any pixel claimed by a higher-priority class is
    removed from lower-priority classes.
    """
    claimed = np.zeros_like(next(iter(masks.values())), dtype=bool)
    resolved: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        m = masks.get(cls)
        if m is None:
            continue
        m = m & ~claimed
        resolved[cls] = m
        claimed = claimed | m
    return resolved


def clean_masks(raw_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Full §6.1 pipeline: per-class cleanup, then priority merge.

    Args:
        raw_masks: {"trunk": HxW bool/0-1 array, "cordon": ..., "cane": ..., "shoot": ...}
    Returns:
        Clean, non-overlapping masks with the same keys.
    """
    cleaned = {cls: clean_class_mask(m) for cls, m in raw_masks.items() if m is not None}
    return merge_by_priority(cleaned)


def combined_binary_mask(clean_masks_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Union of all class masks -> single binary vine mask, input to §6.2 skeletonization."""
    shapes = [m.shape for m in clean_masks_dict.values()]
    h, w = shapes[0]
    combined = np.zeros((h, w), dtype=bool)
    for m in clean_masks_dict.values():
        combined |= m
    return combined


def class_label_map(clean_masks_dict: dict[str, np.ndarray]) -> np.ndarray:
    """H×W int array: 0=background, 1=trunk, 2=cordon, 3=cane, 4=shoot.

    Used later (§6.3 step 4) to label skeleton edges by dominant class along
    the path.
    """
    first = next(iter(clean_masks_dict.values()))
    label_map = np.zeros_like(first, dtype=np.uint8)
    for idx, cls in enumerate(CLASSES, start=1):
        m = clean_masks_dict.get(cls)
        if m is not None:
            label_map[m] = idx
    return label_map
