# Grapevine Structural Tracing Model — V1

Implementation scaffold for `grapevine_prd_v1_rewrite.md`. Every script maps to a
numbered PRD section (noted in each file's docstring) so you can trace code back
to spec.

## File structure

```
grapevine/
├── README.md                       ← you are here
├── requirements.txt                 pip deps
├── configs/
│   └── train_config.yaml            PRD §4.2 training hyperparameters
├── data/
│   ├── raw/                         source images (not tracked, drop images here)
│   ├── annotations/                 YOLO-seg .txt labels, one per image
│   └── splits/                      train.txt / val.txt / test.txt (vine-ID isolated)
├── src/grapevine/                   importable package — the actual pipeline logic
│   ├── __init__.py
│   ├── schema.py                    PRD §8 output JSON structure
│   ├── losses.py                    PRD §4.3 focal-loss patch for seg head
│   ├── postprocess.py               PRD §6.1 mask cleanup
│   ├── skeletonize.py               PRD §6.2 skeletonization
│   ├── graph_build.py               PRD §6.3 graph construction
│   ├── trace.py                     PRD §6.4 whole-vine tracing
│   ├── measurements.py              PRD §7 measurements
│   ├── failure_classification.py    PRD §5.3 F1–F5 bucketing
│   └── visualize.py                 PRD §9 4-panel overlay PNG
├── scripts/                          CLI entry points — run these
│   ├── validate_dataset.py          PRD §3.4 quality gate (run before training)
│   ├── train.py                     PRD §4 training
│   ├── infer.py                     raw YOLO inference → per-class masks
│   └── pipeline.py                  end-to-end: image → JSON + overlay PNG
└── tests/
    └── test_pipeline.py             smoke tests for the postprocess→trace chain
```

## Setup

```bash
cd grapevine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run order

**1. Before annotating/training — validate the dataset against the PRD §3.4 gate:**
```bash
python scripts/validate_dataset.py \
    --images data/raw \
    --labels data/annotations \
    --splits data/splits \
    --classes trunk cordon cane shoot
```
Fails loudly (non-zero exit) if any Section 3.2/3.4 minimum isn't met. Fix the
dataset, don't fix the script.

**2. Train:**
```bash
python scripts/train.py --config configs/train_config.yaml \
    --data data/splits --out runs/train1
```
Writes `runs/train1/weights/best.pt`.

**3. Run the full pipeline on a single image (this is the deliverable — image in, PRD §8 JSON + §9 PNG out):**
```bash
python scripts/pipeline.py \
    --weights runs/train1/weights/best.pt \
    --image data/raw/vine_0231.jpg \
    --out-dir outputs/vine_0231
```
Produces:
```
outputs/vine_0231/
├── result.json     # PRD §8 schema
└── overlay.png      # PRD §9 4-panel visualization
```

**4. Just want raw masks, no graph/tracing?**
```bash
python scripts/infer.py --weights runs/train1/weights/best.pt --image path.jpg --out masks.npz
```

## Notes on what's implemented vs. stubbed

- `postprocess.py`, `skeletonize.py`, `graph_build.py`, `trace.py`,
  `measurements.py`, `visualize.py`, `schema.py` implement the PRD's Section 6–9
  logic directly (morphological cleanup, Zhang-Suen skeletonization, node/edge
  graph construction with spur pruning, root-finding + hierarchy-enforced BFS,
  pixel-length/angle measurement, 4-panel PNG).
- `train.py` / `infer.py` are thin wrappers around `ultralytics.YOLO` matching
  the PRD §4.2 hyperparameter table. The PRD's focal-loss patch (§4.3) is
  implemented in `losses.py` as a monkey-patch since Ultralytics doesn't expose
  a `focal_gamma` flag for the seg head natively — this is called out in the
  PRD itself ("If YOLO doesn't support this natively, patch the loss function.
  Document the patch.") and the patch is documented inline in that file.
- `validate_dataset.py` implements the countable checks from §3.2/§3.4
  (image count, per-class instance counts, license field present, vine-ID
  train/test overlap). Inter-annotator IoU (≥0.85 on 50 overlap images) needs
  two independent annotation sets to compute — the script computes it *if* you
  pass `--annotator-a` / `--annotator-b` label dirs, otherwise it flags that
  check as `SKIPPED: needs two annotator sets`.
