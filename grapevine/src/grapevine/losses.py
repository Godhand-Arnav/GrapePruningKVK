"""PRD §4.3 — Class imbalance handling.

Shoots outnumber trunks ~20:1 in pixel count (§4.3). Ultralytics' stock
v8SegmentationLoss uses BCE for the mask branch with no focal-loss option
exposed via config, so per the PRD's own instruction ("If YOLO doesn't
support this natively, patch the loss function. Document the patch.") we
monkey-patch the segmentation loss term with a focal variant at gamma=2.0.

This is imported (for its side effect) at the top of scripts/train.py,
*before* the model/trainer is constructed, so the patched class is what
Ultralytics resolves at training time.

Patch mechanics:
    - We keep everything in ultralytics.utils.loss.v8SegmentationLoss except
      the per-pixel BCE call inside `single_mask_loss`, which we replace with
      a focal-weighted BCE: FL(p_t) = -(1 - p_t)^gamma * log(p_t).
    - gamma is fixed at 2.0 per the PRD; alpha is left at 1.0 (no additional
      per-class weighting) since the PRD only specifies gamma.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

FOCAL_GAMMA = 2.0
_PATCHED = False


def _focal_bce_with_logits(pred, target, gamma: float = FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = torch.exp(-bce)  # p_t = probability assigned to the true class
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def apply_focal_seg_loss_patch(gamma: float = FOCAL_GAMMA) -> None:
    """Monkey-patch Ultralytics' segmentation mask loss to use focal BCE.

    Idempotent — safe to call multiple times (e.g. if train.py and a notebook
    both import this module).
    """
    global _PATCHED
    if _PATCHED:
        return

    from ultralytics.utils.loss import v8SegmentationLoss

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mirrors Ultralytics' original single_mask_loss signature/logic,
        # swapping the final BCE reduction for focal BCE.
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, h, w)
        loss = _focal_bce_with_logits(pred_mask, gt_mask, gamma=gamma)
        return (loss * area.unsqueeze(1)).mean() if area.dim() else loss

    v8SegmentationLoss.single_mask_loss = single_mask_loss
    _PATCHED = True
    print(f"[losses.py] Patched v8SegmentationLoss.single_mask_loss with focal "
          f"BCE (gamma={gamma}) per PRD §4.3.")
