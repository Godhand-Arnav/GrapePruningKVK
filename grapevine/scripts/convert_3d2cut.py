#!/usr/bin/env python
"""Phase 3 — 3D2cut Single Guyot to YOLO-seg conversion.

Maps 3D2cut skeletal tree annotations to 4-class polygonal segmentation masks:
  0: trunk  (root, mainTrunk, oldWood)
  1: cordon (courson - Guyot renewal spur / perennial structure)
  2: cane   (cane)
  3: shoot  (shoot, lateralShoot)

Outputs:
  data/
  ├── annotations/             (all YOLO-seg labels for validate_dataset.py)
  ├── images/{train,val,test}/
  ├── labels/{train,val,test}/
  ├── splits/{train.txt, val.txt, test.txt, data.yaml}
  └── raw/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml

LABEL_MAP = {
    "root": 0,
    "mainTrunk": 0,
    "oldWood": 0,
    "courson": 1,
    "cane": 2,
    "shoot": 3,
    "lateralShoot": 3,
}

# Ribbon radius in native 4032x3024 pixels
# Produces realistic branch thickness and survives PRD §6.1 opening (r=2 at 640)
RADIUS_MAP = {
    0: 28,  # trunk ~56px
    1: 18,  # cordon/courson ~36px
    2: 14,  # cane ~28px
    3: 8,   # shoot ~16px
}

CLASS_NAMES = {0: "trunk", 1: "cordon", 2: "cane", 3: "shoot"}


def safe_link_or_copy(src: Path, dst: Path) -> None:
    """Link file without duplicating data if possible, fallback to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # Hard link on Windows/Linux (fast, 0 extra disk space)
    except (OSError, NotImplementedError):
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def convert_annotation_to_polygons(json_path: Path) -> tuple[str, int, int, list[tuple[int, list[float]]]]:
    """Reads 3D2cut JSON and returns (image_filename, width, height, [(cls_id, [x1, y1, ...]), ...])."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vimg = data["VineImage"][0]
    img_filename = vimg["ImageFileName"]
    feats = vimg.get("VineFeature", [])

    feat_list = []
    if feats and isinstance(feats[0], list):
        for sub in feats:
            feat_list.extend(sub)
    else:
        feat_list = feats

    id_to_feat = {ft["FeatureID"]: ft for ft in feat_list}

    # Inspect image resolution (default 4032x3024)
    w, h = 4032, 3024
    for ft in feat_list:
        coords = ft.get("FeatureCoordinates")
        if coords:
            if coords[0] > w:
                w = int(coords[0]) + 100
            if coords[1] > h:
                h = int(coords[1]) + 100

    masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(4)}

    for ft in feat_list:
        pid = ft.get("ParentID")
        if pid is None or pid not in id_to_feat:
            continue
        p_ft = id_to_feat[pid]
        c1 = tuple(map(int, ft["FeatureCoordinates"]))
        c2 = tuple(map(int, p_ft["FeatureCoordinates"]))
        lbl = ft.get("BranchLabel")
        cls_id = LABEL_MAP.get(lbl)
        if cls_id is None:
            continue

        r = RADIUS_MAP[cls_id]
        cv2.line(masks[cls_id], c1, c2, 1, thickness=r * 2)
        cv2.circle(masks[cls_id], c1, r, 1, -1)
        cv2.circle(masks[cls_id], c2, r, 1, -1)

    # Convert binary class masks to YOLO normalized polygons
    polygons = []
    for cls_id in range(4):
        mask = masks[cls_id]
        if not np.any(mask):
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # ignore tiny speckles
                continue

            # Approximate contour to reduce points while preserving shape
            epsilon = 1.2
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            if len(approx) < 3:
                continue

            pts_norm = []
            for pt in approx.reshape(-1, 2):
                nx = max(0.0, min(1.0, float(pt[0]) / w))
                ny = max(0.0, min(1.0, float(pt[1]) / h))
                pts_norm.extend([round(nx, 6), round(ny, 6)])

            if len(pts_norm) >= 6:  # at least 3 points
                polygons.append((cls_id, pts_norm))

    return img_filename, w, h, polygons


def main():
    parser = argparse.ArgumentParser(description="Convert 3D2cut Single Guyot dataset to YOLO-seg")
    parser.add_argument("--root", type=Path, default=Path("data/3D2cut_Single_Guyot"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    train_val_dir = args.root / "01-TrainAndValidationSet"
    test_dir = args.root / "02-IndependentTestSet"

    if not train_val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing 3D2cut folders in {args.root}")

    # Output paths
    annotations_dir = args.output / "annotations"
    raw_dir = args.output / "raw"
    splits_dir = args.output / "splits"
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"

    for d in [annotations_dir, raw_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        (images_dir / s).mkdir(parents=True, exist_ok=True)
        (labels_dir / s).mkdir(parents=True, exist_ok=True)

    # 1. Process Test Set (02-IndependentTestSet)
    test_json_files = sorted(test_dir.glob("*_annotation.json"))
    print(f"Processing {len(test_json_files)} test set files...")

    test_filenames = []
    manifest_rows = []

    for jf in test_json_files:
        stem = jf.name.replace("_annotation.json", "")
        img_candidates = list(test_dir.glob(f"{stem}.*"))
        img_candidates = [p for p in img_candidates if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not img_candidates:
            continue
        img_path = img_candidates[0]
        img_fname = img_path.name
        test_filenames.append(img_fname)

        vine_id = f"test_{stem}"
        manifest_rows.append({
            "filename": img_fname,
            "vine_id": vine_id,
            "license": "CC BY-NC-SA 4.0",
        })

        safe_link_or_copy(img_path, raw_dir / img_fname)
        safe_link_or_copy(img_path, images_dir / "test" / img_fname)

        _, _, _, polygons = convert_annotation_to_polygons(jf)

        lbl_lines = [f"{cls_id} " + " ".join(map(str, pts)) for cls_id, pts in polygons]
        lbl_content = "\n".join(lbl_lines) + "\n" if lbl_lines else ""

        txt_name = f"{img_path.stem}.txt"
        (annotations_dir / txt_name).write_text(lbl_content, encoding="utf-8")
        (labels_dir / "test" / txt_name).write_text(lbl_content, encoding="utf-8")

    # 2. Process Train & Val Set (01-TrainAndValidationSet)
    train_val_json_files = sorted(train_val_dir.glob("*_annotation.json"))
    print(f"Processing {len(train_val_json_files)} train/val set files...")

    train_filenames = []
    val_filenames = []

    for i, jf in enumerate(train_val_json_files):
        stem = jf.name.replace("_annotation.json", "")
        img_candidates = list(train_val_dir.glob(f"{stem}.*"))
        img_candidates = [p for p in img_candidates if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not img_candidates:
            continue
        img_path = img_candidates[0]
        img_fname = img_path.name

        vine_id = f"train_{stem}"
        manifest_rows.append({
            "filename": img_fname,
            "vine_id": vine_id,
            "license": "CC BY-NC-SA 4.0",
        })

        safe_link_or_copy(img_path, raw_dir / img_fname)

        is_val = (i % int(1.0 / args.val_ratio)) == 0 if args.val_ratio > 0 else False
        split = "val" if is_val else "train"

        if is_val:
            val_filenames.append(img_fname)
        else:
            train_filenames.append(img_fname)

        safe_link_or_copy(img_path, images_dir / split / img_fname)

        _, _, _, polygons = convert_annotation_to_polygons(jf)
        lbl_lines = [f"{cls_id} " + " ".join(map(str, pts)) for cls_id, pts in polygons]
        lbl_content = "\n".join(lbl_lines) + "\n" if lbl_lines else ""

        txt_name = f"{img_path.stem}.txt"
        (annotations_dir / txt_name).write_text(lbl_content, encoding="utf-8")
        (labels_dir / split / txt_name).write_text(lbl_content, encoding="utf-8")

    # 3. Write manifest.csv to raw and images
    for m_dir in [raw_dir, images_dir]:
        with open(m_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "vine_id", "license"])
            writer.writeheader()
            writer.writerows(manifest_rows)

    # 4. Write splits txt files
    (splits_dir / "train.txt").write_text("\n".join(train_filenames) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_filenames) + "\n", encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(test_filenames) + "\n", encoding="utf-8")

    # 5. Write data.yaml
    data_yaml = {
        "path": str(args.output.resolve()),
        "train": str((images_dir / "train").resolve()),
        "val": str((images_dir / "val").resolve()),
        "test": str((images_dir / "test").resolve()),
        "names": CLASS_NAMES,
    }
    with open(splits_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print("Conversion completed successfully:")
    print(f"  Train samples: {len(train_filenames)}")
    print(f"  Val samples:   {len(val_filenames)}")
    print(f"  Test samples:  {len(test_filenames)}")
    print(f"  Labels written to: {annotations_dir} and {labels_dir}")
    print(f"  Manifest written to: {raw_dir / 'manifest.csv'}")
    print(f"  Data YAML written to: {splits_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
