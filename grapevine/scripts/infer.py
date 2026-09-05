#!/usr/bin/env python
"""Raw YOLO-seg inference -> per-class binary masks (H x W, 640x640 per §2).

This is the first stage of the pipeline (before postprocess/skeletonize/
graph/trace). Use pipeline.py for the full image -> JSON + PNG flow; use this
directly if you just want masks (e.g. for debugging the model in isolation).

Usage:
    python scripts/infer.py --weights runs/train1/weights/best.pt \
        --image data/raw/vine_0231.jpg --out masks.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

CLASSES = ["trunk", "cordon", "cane", "shoot"]


def run_inference(weights: str, image_path: str) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Returns (masks_by_class, original_image_rgb, inference_ms)."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(weights)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    t0 = time.perf_counter()
    results = model.predict(source=image_path, imgsz=640, verbose=False)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    h, w = img.shape[:2]
    masks_by_class = {cls: np.zeros((h, w), dtype=bool) for cls in CLASSES}

    if results.masks is not None:
        mask_data = results.masks.data.cpu().numpy()  # (N, Hm, Wm)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        for m, cid in zip(mask_data, cls_ids):
            if cid >= len(CLASSES):
                continue
            m_resized = _resize_mask(m, (h, w))
            masks_by_class[CLASSES[cid]] |= m_resized.astype(bool)

    return masks_by_class, img, inference_ms


def _resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    import cv2
    return cv2.resize(mask.astype(np.uint8), (target_hw[1], target_hw[0]),
                       interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    masks, img, ms = run_inference(args.weights, args.image)
    np.savez_compressed(args.out, image=img, inference_ms=ms, **masks)
    print(f"Inference: {ms:.1f}ms. Saved masks for classes {list(masks.keys())} to {args.out}")


if __name__ == "__main__":
    main()
