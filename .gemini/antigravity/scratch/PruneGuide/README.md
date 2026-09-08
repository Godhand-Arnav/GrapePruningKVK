# PruneGuide: AI-Based Grapevine Pruning Decision Support

PruneGuide is a lightweight 2D deep-learning vision system for assisting non-expert vineyard labor with winter pruning decisions using a **Record-Now, Prune-Later** workflow.

The architecture combines:
1. **2-Stage Stacked Hourglass Network (SHG)**: Detects organ landmarks (Trunk, Courson, Cane, Shoot nodes) and branch Part Affinity Fields (PAFs).
2. **Resistivity Graph Optimization (Dijkstra)**: Reconstructs the 2D plant skeleton tree from the Root crown outwards without requiring 3D sensors or complex AR tracking.
3. **Agronomic Pruning Rule Engine (Simonit & Sirch)**: Ranks renewal spurs vs fruiting canes, checks bud counts, and marks precise 2D cutting locations.

---

## 1. Quick Start in Antigravity IDE

### Open Workspace
In Antigravity IDE, open the project folder:
C:\Users\shrus\.gemini\antigravity\scratch\PruneGuide

### (Optional) Enable NVIDIA RTX 4050 CUDA Acceleration
To enable GPU acceleration instead of CPU for deep training runs, install PyTorch with CUDA:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

---

## 2. Testing the Pipeline Immediately (Dry Run)

A synthetic dataset generator is included so you can test data loading, training, and inference without waiting for large dataset downloads.

### Step A: Generate Synthetic Grapevine Data
python scripts/generate_synthetic.py
This generates 12 training samples and 4 test samples in data/samples/.

### Step B: Train the Model
python scripts/train.py --epochs 5 --img_size 512 --hg_channels 64
* The best model checkpoint is automatically saved to checkpoints/best_model.pth.

### Step C: Run Inference & Pruning Visual Decision Support
python scripts/infer.py --img_size 512
* Output annotated image with color-coded cuts is saved to:
  outputs/annotated_pruning_decision.png

---

## 3. Training on the Real 3D2cut Dataset

1. Download the **3D2cut Single Guyot dataset** from the official repository:
   * DOI: [10.34777/azf6-tm83](https://doi.org/10.34777/azf6-tm83)
2. Extract the dataset into data/raw/ so the directory structure is:
   `
   data/raw/
   ├── train/
   │   ├── images/       (*.jpg, *.png)
   │   └── annotations/  (*.json)
   └── test/
       ├── images/       (*.jpg, *.png)
       └── annotations/  (*.json)
   `
3. Run training on your GPU:
   python scripts/train.py --data_dir data/raw --epochs 100 --img_size 1024 --hg_channels 128 --batch_size 1

---

## 4. Project Structure

`
PruneGuide/
├── checkpoints/             # Saved model weights (best_model.pth)
├── data/
│   ├── raw/                 # Real 3D2cut dataset
│   └── samples/             # Synthetic samples for testing
├── models/
│   ├── blocks.py            # Residual sub-blocks & instance normalization
│   └── hourglass.py         # 2-Stage Stacked Hourglass Network
├── outputs/                 # Annotated decision support output images
├── pipeline/
│   ├── dataset.py           # Gaussian heatmap and PAF vector field generator
│   ├── graph_extractor.py   # Peak extraction & Dijkstra shortest path tree builder
│   └── rule_engine.py       # Simonit & Sirch pruning rules & visual overlay
├── scripts/
│   ├── download_dataset.py  # 3D2cut dataset setup instructions
│   ├── generate_synthetic.py# Synthetic vine dataset generator
│   ├── infer.py             # Inference pipeline: Image -> Tree -> Cut Points
│   └── train.py             # Model training script
└── requirements.txt         # Dependencies
`
