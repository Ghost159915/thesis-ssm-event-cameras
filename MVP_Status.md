# MVP Setup Status
## Benas Vaiciulis — UNSW MMAN4951 | Last updated: April 2026

---

## Current Status: 🟡 IN PROGRESS — PyTorch Installing

---

## Environment

| Item | Status | Detail |
|------|--------|--------|
| OS | ✅ Done | Ubuntu 24.04 LTS (Noble) |
| GPU | ✅ Done | AMD Radeon RX 7700 XT (12 GB VRAM) |
| ROCm | ✅ Done | ROCm 6.4.2 installed |
| HSA override | ✅ Done | `HSA_OVERRIDE_GFX_VERSION=11.0.0` in `~/.bashrc` |
| Conda | ✅ Done | Miniforge installed |
| Conda env | ✅ Done | `events_signals` (Python 3.11) — use this, NOT `ssm_thesis` |
| PyTorch | 🟡 Installing | `torch==2.3.0+rocm6.0` — 2.2 GB download in progress |
| Repo | ✅ Done | Cloned to `~/Documents/Thesis/ssms_event_cameras` |
| Gen1 dataset | 🟡 In progress | Downloading `gen1.tar` to `~/Documents/Thesis/data/gen1_raw` |

---

## Key Corrections Found (not in original guide)

| Issue | Fix |
|-------|-----|
| `requirements.txt` is NOT at repo root | It is at `RVT/requirements.txt` |
| Repo uses its own install instructions | Follow `installation_details.txt` in repo root, NOT the MVP_Setup_Guide_Complete.md pip line for PyTorch |
| `pytorch-cuda=11.8` in `installation_details.txt` is for NVIDIA | Replace with: `pip install torch==2.3.0+rocm6.0 torchvision==0.15.2+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0` |
| torch==2.2.1 not available on ROCm 6.0 index | Use `torch==2.3.0+rocm6.0` instead |

---

## Completed Steps

- [x] Ubuntu 24.04 installed and updated
- [x] ROCm 6.4.2 installed
- [x] User added to `render` and `video` groups
- [x] `HSA_OVERRIDE_GFX_VERSION=11.0.0` added to `~/.bashrc`
- [x] Miniforge (conda) installed
- [x] Conda environment `events_signals` created (Python 3.11)
- [x] Repo cloned: `~/Documents/Thesis/ssms_event_cameras`
- [x] Gen1 dataset download started to `~/Documents/Thesis/data/gen1_raw`

---

## Next Steps (in order)

### Step 1 — Verify PyTorch installed correctly
```bash
conda activate events_signals
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Expected: `2.3.0+rocm6.0` and `True`

If `False`: verify `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set (`echo $HSA_OVERRIDE_GFX_VERSION`)

### Step 2 — Install remaining packages (from installation_details.txt)
```bash
conda activate events_signals
cd ~/Documents/Thesis/ssms_event_cameras
pip install lightning wandb pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn
```

### Step 3 — Install S5 and RVT packages
```bash
cd ~/Documents/Thesis/ssms_event_cameras
pip install -e ./modules/s5
pip install -e ./rpg_rvt/
```

Verify:
```bash
python -c "from s5 import S5; print('S5 OK')"
```

### Step 4 — Extract and preprocess Gen1 dataset
Once `gen1.tar` finishes downloading:
```bash
cd ~/Documents/Thesis/data/gen1_raw
tar -xf gen1.tar

# Check structure — should show test/ train/ val/ subdirectories
ls -lh

# Preprocess to HDF5 voxel format
cd ~/Documents/Thesis/ssms_event_cameras
python RVT/preprocess/gen1.py \
  --data_dir ~/Documents/Thesis/data/gen1_raw \
  --output_dir ~/Documents/Thesis/data/gen1_processed \
  --num_workers 4
```

### Step 5 — Download pretrained checkpoint
```bash
mkdir -p ~/Documents/Thesis/checkpoints
wget -O ~/Documents/Thesis/checkpoints/s5_gen1_base.ckpt \
  https://download.ifi.uzh.ch/rpg/ssms_event_cameras/checkpoints/gen1/s5_vitb_gen1.ckpt
ls -lh ~/Documents/Thesis/checkpoints/s5_gen1_base.ckpt
# Should be ~400 MB
```

### Step 6 — Run evaluation
```bash
conda activate events_signals
cd ~/Documents/Thesis/ssms_event_cameras
HSA_OVERRIDE_GFX_VERSION=11.0.0 python RVT/validation.py \
  dataset=gen1 \
  dataset.path=~/Documents/Thesis/data/gen1_processed \
  checkpoint=~/Documents/Thesis/checkpoints/s5_gen1_base.ckpt \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001 \
  2>&1 | tee ~/Documents/Thesis/results/gen1_s5_vitb_eval.txt
```

---

## Important Notes

- **Always activate environment first:** `conda activate events_signals`
- **Always set HSA override:** `export HSA_OVERRIDE_GFX_VERSION=11.0.0` (or source ~/.bashrc)
- **Working directory for eval:** must be `~/Documents/Thesis/ssms_event_cameras`
- **Data paths use `~/Documents/Thesis/`** — different from the setup guide which used `~/data/`
- **Conda env name is `events_signals`** — not `ssm_thesis` as in the setup guide

---

## Target Result

| Model | Dataset | mAP@0.5 | Reference |
|-------|---------|---------|-----------|
| S5-RVT (ViT-Base) | Gen1 test | **47.71** | Zubic et al. CVPR 2024 |

Once evaluation completes, update `main.tex` placeholders:
- `[RESULT_MAP]` → overall mAP@0.5
- `[RESULT_CAR]` → car AP@0.5
- `[RESULT_PED]` → pedestrian AP@0.5

---

*Resume from this document in a new session — all paths and corrections are captured above.*
