#!/usr/bin/env python
"""PRD §3.4 — Dataset Quality Gate. Run before annotation is considered final
and before training starts.

Checks:
  [ ] >=500 images pass audit                          (§3.4)
  [ ] Per-class counts meet §3.2 minimums (train + test) (§3.2)
  [ ] Vine IDs assigned, no train/test vine overlap      (§3.4, §2)
  [ ] License field present per image                    (§3.4)
  [ ] Inter-annotator IoU >=0.85 on 50 overlap images     (§3.4) — only if
      --annotator-a / --annotator-b are provided; otherwise reported SKIPPED.

Exits non-zero if any check fails. Per the PRD: "Gatekeeper: Project lead
signs off. No exceptions." — this script doesn't sign off for you, it just
tells you whether you're allowed to ask.

Expected layout:
    <images>/*.jpg
    <labels>/*.txt          YOLO-seg format, one label file per image
    <splits>/train.txt      one image filename per line
    <splits>/test.txt
    A sidecar <images>/manifest.csv with columns: filename,vine_id,license
    is required for the vine-overlap and license checks.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

MIN_TOTAL_IMAGES = 500
MIN_INSTANCES = {"trunk": 400, "cordon": 350, "cane": 500, "shoot": 800}
MIN_TEST_INSTANCES = {"trunk": 40, "cordon": 35, "cane": 50, "shoot": 80}
IOU_THRESHOLD = 0.85


def load_manifest(images_dir: Path) -> dict[str, dict]:
    manifest_path = images_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    rows = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["filename"]] = row
    return rows


def count_instances(labels_dir: Path, filenames: list[str], classes: list[str]) -> Counter:
    counts = Counter()
    for fname in filenames:
        label_path = labels_dir / (Path(fname).stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_idx = int(line.split()[0])
                if 0 <= cls_idx < len(classes):
                    counts[classes[cls_idx]] += 1
    return counts


def read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="PRD §3.4 dataset quality gate")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--classes", nargs=4, required=True,
                     help="class names in label-index order, e.g. trunk cordon cane shoot")
    ap.add_argument("--annotator-a", type=Path, default=None,
                     help="optional: label dir from annotator A, for IoU agreement check")
    ap.add_argument("--annotator-b", type=Path, default=None,
                     help="optional: label dir from annotator B, for IoU agreement check")
    args = ap.parse_args()

    classes = args.classes
    failures: list[str] = []
    warnings: list[str] = []

    # --- image count ---
    all_images = sorted(p.name for p in args.images.glob("*.jpg")) + \
                 sorted(p.name for p in args.images.glob("*.png"))
    print(f"[check] total images found: {len(all_images)}")
    if len(all_images) < MIN_TOTAL_IMAGES:
        failures.append(f">=500 images required, found {len(all_images)}")

    # --- per-class instance counts (whole dataset) ---
    all_counts = count_instances(args.labels, all_images, classes)
    print(f"[check] instance counts (dataset): {dict(all_counts)}")
    for cls, min_count in MIN_INSTANCES.items():
        if cls not in classes:
            continue
        if all_counts[cls] < min_count:
            failures.append(f"class '{cls}': {all_counts[cls]} instances, need >= {min_count} (§3.2)")

    # --- per-class instance counts (test split) ---
    test_split = read_split(args.splits / "test.txt")
    if not test_split:
        failures.append("data/splits/test.txt missing or empty")
    else:
        test_counts = count_instances(args.labels, test_split, classes)
        print(f"[check] instance counts (test): {dict(test_counts)}")
        for cls, min_count in MIN_TEST_INSTANCES.items():
            if cls not in classes:
                continue
            if test_counts[cls] < min_count:
                failures.append(f"test class '{cls}': {test_counts[cls]} instances, need >= {min_count} (§3.2)")

    # --- vine ID train/test isolation + license ---
    manifest = load_manifest(args.images)
    if not manifest:
        warnings.append("no manifest.csv found — cannot verify vine-ID isolation or license (§3.4)")
    else:
        train_split = read_split(args.splits / "train.txt")
        train_vines = {manifest[f]["vine_id"] for f in train_split if f in manifest}
        test_vines = {manifest[f]["vine_id"] for f in test_split if f in manifest}
        overlap = train_vines & test_vines
        if overlap:
            failures.append(f"train/test vine ID overlap detected: {sorted(overlap)} — zero leakage required (§2)")
        else:
            print("[check] vine ID isolation: OK, no overlap")

        missing_license = [f for f in all_images if f in manifest and not manifest[f].get("license")]
        if missing_license:
            failures.append(f"{len(missing_license)} images missing license field (§3.4)")
        else:
            print("[check] license field: present for all manifested images")

    # --- inter-annotator IoU ---
    if args.annotator_a and args.annotator_b:
        iou = _mean_iou(args.annotator_a, args.annotator_b, classes)
        print(f"[check] inter-annotator mean IoU: {iou:.3f}")
        if iou < IOU_THRESHOLD:
            failures.append(f"inter-annotator IoU {iou:.3f} < {IOU_THRESHOLD} (§3.4)")
    else:
        print("[check] inter-annotator IoU: SKIPPED (needs --annotator-a and --annotator-b)")

    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print("FAILED — dataset does not pass the §3.4 quality gate:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASSED all automatable §3.4 checks. Project lead sign-off still required.")


def _mean_iou(dir_a: Path, dir_b: Path, classes: list[str]) -> float:
    """Rough per-class-agnostic pixel IoU between two YOLO-seg label sets on
    their shared filenames. This is a lightweight proxy for the annotation
    QA step, not a substitute for a proper polygon-IoU tool."""
    import numpy as np

    shared = sorted(set(p.stem for p in dir_a.glob("*.txt")) & set(p.stem for p in dir_b.glob("*.txt")))
    if not shared:
        return 0.0
    ious = []
    for stem in shared:
        a = _labels_to_binary(dir_a / f"{stem}.txt")
        b = _labels_to_binary(dir_b / f"{stem}.txt")
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _labels_to_binary(path: Path, size: int = 640):
    import cv2
    import numpy as np
    mask_u8 = np.zeros((size, size), dtype=np.uint8)
    if not path.exists():
        return mask_u8.astype(bool)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2) * size
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask_u8, [pts], 1)
    return mask_u8.astype(bool)


if __name__ == "__main__":
    main()
