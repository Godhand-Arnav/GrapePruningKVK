#!/usr/bin/env python
"""PRD §4 — Model training.

Thin wrapper around ultralytics.YOLO that:
  - applies the §4.3 focal-loss patch before model construction
  - loads all hyperparameters from configs/train_config.yaml (§4.2)
  - schedules mosaic augmentation to close at epoch 150 (§4.2)

Usage:
    python scripts/train.py --config configs/train_config.yaml \
        --data data/splits --out runs/train1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grapevine.losses import apply_focal_seg_loss_patch  # noqa: E402  (must precede YOLO import use)


def build_data_yaml(splits_dir: Path, classes: dict[int, str], out_path: Path) -> Path:
    data_cfg = {
        "path": str(splits_dir.parent.resolve()),
        "train": str((splits_dir / "train.txt").resolve()),
        "val": str((splits_dir / "val.txt").resolve()),
        "test": str((splits_dir / "test.txt").resolve()),
        "names": classes,
    }
    out_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    return out_path


def main():
    ap = argparse.ArgumentParser(description="PRD §4 training entry point")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="dir containing train.txt/val.txt/test.txt")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    # PRD §4.3 — apply focal loss patch BEFORE constructing the model
    apply_focal_seg_loss_patch(gamma=2.0)

    from ultralytics import YOLO  # imported after the patch is applied

    data_yaml = build_data_yaml(args.data, cfg["classes"], args.out.parent / "data.yaml")

    model = YOLO(cfg["model"])

    # §4.2: mosaic p=1.0 for first 150 epochs, then 0 -> use Ultralytics' close_mosaic
    close_mosaic = cfg.get("mosaic_close_epoch", 150)

    model.train(
        data=str(data_yaml),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch=cfg["batch"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        momentum=cfg["momentum"],
        box=cfg["box"],
        cls=cfg["cls"],
        dfl=cfg["dfl"],
        # note: 'seg' loss weight (cfg["seg"]) is baked into the focal patch's
        # weighting rather than passed here, since Ultralytics computes the
        # seg loss weight internally in v8SegmentationLoss.__call__.
        fliplr=cfg["fliplr"],
        scale=cfg["scale"],
        translate=cfg["translate"],
        mosaic=cfg["mosaic"],
        close_mosaic=close_mosaic,
        degrees=cfg["degrees"],   # 0.0 — no rotation, per §4.2
        hsv_h=cfg["hsv_h"],       # 0.0 — no color jitter
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        project=str(args.out.parent),
        name=args.out.name,
    )


if __name__ == "__main__":
    main()
