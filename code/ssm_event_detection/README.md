# SSM Event Camera Object Detector

SSM-based object detection on event camera data. Two model architectures for direct comparison:
- **Hybrid** — CNN backbone (ResNet-18) + Mamba temporal processing
- **Pure SSM** — No CNN. Patch embedding + Bidirectional Mamba (spatial) + Mamba (temporal)

**Thesis:** Event-Based Vision and State Space Models for Object Detection and Autonomous Obstacle Avoidance on Micro UAVs — Benas Vaiciulis, UNSW 2026.

---

## Models

### Model 1: CNN-SSM Hybrid (`configs/default.yaml`)

```
Voxel grid (B, T, H, W)
    │
    ├── Shared ResNet-18 backbone per temporal bin  → (B, T, 256, H/8, W/8)
    ├── Global avg pool → sequence tokens           → (B, T, 256)
    ├── Mamba temporal stack (2 layers)             → (B, T, 256)
    ├── Fuse last-token context → spatial map
    └── Anchor-free detection head
```

Spatial features extracted by CNN, temporal reasoning by Mamba.
Baseline model — directly comparable to CNN+RNN and CNN+Transformer approaches.

### Model 2: Pure SSM (`configs/pure_ssm.yaml`)

```
Voxel grid (B, T, H, W)
    │
    ├── Patch embedding (16×16 patches, no convolution) → (B, T, num_patches, 256)
    ├── Bidirectional Mamba — spatial (per bin)         → (B, T, num_patches, 256)
    ├── Pool → Mamba temporal (across bins)             → (B, T, 256)
    ├── Fuse temporal context → patch token grid
    └── Anchor-free detection head
```

No CNN anywhere. Mamba handles both spatial and temporal processing.
**Novel contribution** — not previously demonstrated for event camera detection.

Both models use the same pure-PyTorch Mamba implementation — no custom CUDA kernels,
so they run on AMD GPUs (ROCm), Apple Silicon (MPS), and NVIDIA (CUDA) without changes.

---

## Setup

### 1. Activate your virtual environment

```bash
source ~/thesis/bin/activate
cd ~/Desktop/Thesis/code/ssm_event_detection
```

### 2. Install dependencies

```bash
pip3 install torch torchvision einops pyyaml tqdm numpy scipy h5py pycocotools matplotlib opencv-python
```

**On macOS M-series:** standard pip install above — MPS is auto-detected.

**On AMD GPU (Windows/Linux with ROCm):**
Install ROCm-enabled PyTorch from https://pytorch.org — select ROCm as compute platform.

**On UNSW Katana (NVIDIA):**
```bash
module load python/3.10 cuda/11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install einops pyyaml tqdm numpy scipy h5py pycocotools
```

### 3. Download the Gen1 dataset

The Gen1 dataset is freely available from Prophesee:

1. Go to: https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
2. Register (free) and download the train/test splits (~40 GB total).
3. Extract to `data/gen1/`:

```
data/gen1/
├── train/
│   ├── <recording>_bbox.npy
│   ├── <recording>_td.npy
│   └── ...
└── test/
    ├── <recording>_bbox.npy
    ├── <recording>_td.npy
    └── ...
```

**Converting .dat → .npy (recommended for 10× faster loading):**
```python
from metavision_core.event_io import EventsIterator
import numpy as np

reader = EventsIterator("recording_td.dat")
events = np.concatenate([
    np.column_stack([ev["x"], ev["y"], ev["t"], ev["p"]]) for ev in reader
])
np.save("recording_td.npy", events)
```

---

## Usage

### Smoke test — no dataset needed

Verifies the full pipeline (data → model → loss → metrics) on synthetic random data.

```bash
# Test hybrid model
python3 train.py --smoke-test

# Test pure SSM model
python3 train.py --config configs/pure_ssm.yaml --smoke-test
```

Expected output:
```
[device] Using MPS          ← or CUDA / CPU
Model: hybrid               ← or pure_ssm
Model parameters: 2.2M
Epoch 1/3 | train_loss: ... | val_loss: ... | mAP: ...
```

### Train on Gen1

```bash
# Train hybrid model
python3 train.py --config configs/default.yaml

# Train pure SSM model
python3 train.py --config configs/pure_ssm.yaml
```

### Train with overrides

```bash
python3 train.py --config configs/pure_ssm.yaml --batch-size 4 --epochs 50 --lr 5e-4
```

### Resume from checkpoint

```bash
python3 train.py --config configs/pure_ssm.yaml --resume outputs/pure_ssm/last.pth
```

---

## Comparing the two models

Run both to completion on Gen1, then compare:

| Metric | Hybrid | Pure SSM |
|--------|--------|----------|
| mAP@0.5 | — | — |
| AP_car | — | — |
| AP_pedestrian | — | — |
| Parameters | ~2.2M | ~2.1M |
| Inference time | — | — |

Fill in after training. This table is the core comparison result for the thesis.

---

## Key hyperparameters

**Hybrid** (`configs/default.yaml`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_bins` | 10 | Temporal bins in voxel grid |
| `mamba.d_state` | 16 | SSM state size — try 32 for better accuracy |
| `mamba.num_layers` | 2 | Mamba blocks |
| `training.batch_size` | 8 | Reduce to 4 on local GPU |
| `training.lr` | 1e-3 | AdamW LR |

**Pure SSM** (`configs/pure_ssm.yaml`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `patch_size` | 16 | 16→285 patches, 8→1140 patches (slower, finer) |
| `spatial_layers` | 2 | BiMamba layers for spatial processing |
| `temporal_layers` | 2 | Causal Mamba layers across bins |
| `training.lr` | 5e-4 | Slightly lower than hybrid |

---

## Output

Each model saves checkpoints to its own folder:
- Hybrid → `outputs/best.pth`, `outputs/last.pth`
- Pure SSM → `outputs/pure_ssm/best.pth`, `outputs/pure_ssm/last.pth`

Metrics reported each epoch:
- `mAP` — mean Average Precision @ IoU 0.5
- `AP_car`, `AP_pedestrian` — per-class AP
- `train_loss`, `val_loss` — combined focal + GIoU loss

---

## Upgrading to official Mamba CUDA kernels (Katana only)

On a CUDA machine, swap in the optimised triton kernels for ~3-5× faster training:

```bash
pip install mamba-ssm
```

Then in `models/detector.py` and `models/pure_ssm_detector.py`, replace:
```python
from .mamba import MambaStack
```
with the `mamba_ssm` equivalent. The pure-PyTorch version is functionally identical
and the model weights are compatible.

---

## Project Structure

```
ssm_event_detection/
├── configs/
│   ├── default.yaml          # Hybrid model config
│   └── pure_ssm.yaml         # Pure SSM model config
├── data/
│   ├── event_representations.py  # Voxel grid (bilinear temporal interp)
│   ├── gen1_dataset.py           # Gen1 dataset loader (.npy and .dat)
│   └── __init__.py
├── models/
│   ├── mamba.py              # Pure-PyTorch Mamba (MambaBlock, MambaStack)
│   ├── detector.py           # CNN-SSM hybrid: EventSSMDetector
│   ├── pure_ssm_detector.py  # Pure SSM: PureSSMDetector (BiMamba spatial + Mamba temporal)
│   └── __init__.py
├── utils/
│   ├── device.py             # Auto device detection (CUDA / ROCm / MPS / CPU)
│   ├── loss.py               # Focal loss + GIoU box loss
│   ├── metrics.py            # mAP@0.5 with per-class AP
│   └── __init__.py
├── outputs/                  # Hybrid model checkpoints
├── train.py                  # Training entry point (handles both models)
├── requirements.txt
└── README.md
```
