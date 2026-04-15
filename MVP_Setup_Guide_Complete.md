# Complete MVP Setup Guide — From Zero to Evaluation
## Ubuntu 24.04 LTS (Noble) | Fresh System Installation
### Thesis B — Benas Vaiciulis | UNSW MMAN4951

**Purpose:** This guide provides a complete, end-to-end setup for reproducing Zubic et al. (2024) on a fresh Ubuntu 24.04 system with AMD RX 7700 XT GPU. It covers system dependencies, GPU drivers, Python environment, PyTorch, all project requirements, dataset preparation, and evaluation.

**Total time:** ~4–6 hours (mostly dataset download/preprocessing)

---

## Part 1: System Dependencies & Updates

### 1.1 Update System Packages

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y build-essential git curl wget gnupg2 cmake
```

### 1.2 Install Kernel Headers (Required for ROCm)

```bash
sudo apt install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
```

---

## Part 2: AMD GPU & ROCm 6.4.2 Setup

### 2.1 Install ROCm Repository & Drivers

Ubuntu 24.04 (Noble) uses a new GPG key format. Follow this exactly:

```bash
# Create keyrings directory
sudo mkdir -p /etc/apt/keyrings

# Download and add ROCm GPG key (to new keyrings location, not trusted.gpg.d)
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add ROCm 6.4.2 repository — **note: "noble" not "jammy"**
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.2 noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install -y rocm-hip-sdk rocm-opencl-sdk rocm-dev rocm-libs
```

### 2.2 Add User to GPU Access Groups

```bash
sudo usermod -aG render,video $USER
```

**Important:** Log out and back in (or restart) for group changes to take effect.

### 2.3 Verify GPU Detection

After restart:

```bash
rocm-smi
```

Expected output:
```
======================= ROCm System Management Interface =======================
GPU[0]  : AMD Radeon RX 7700 XT (gfx1102)
GPU[0]  : Temp: 45°C | Power: 15W | VRAM Used: 0MB / 12288MB
```

If GPU is not detected:
```bash
lsmod | grep amdgpu
# Should show: amdgpu kernel module loaded
```

### 2.4 Set RDNA3 GFX Override (Critical for RX 7700 XT)

Add to `~/.bashrc`:

```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
echo $HSA_OVERRIDE_GFX_VERSION
# Expected: 11.0.0
```

---

## Part 3: Python & Conda Environment

### 3.1 Install Miniforge (Lightweight Conda)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# Follow installer prompts:
# - Press Enter to accept location
# - Type 'yes' to initialize conda
```

After installation:
```bash
source ~/.bashrc
```

### 3.2 Create Python Environment

```bash
conda create -n ssm_thesis python=3.10 -y
conda activate ssm_thesis
```

This environment will be your workspace for all thesis work.

---

## Part 4: PyTorch with ROCm Support

### 4.1 Install PyTorch ROCm Wheels

Use ROCm 6.0 wheels (closest official support to ROCm 6.4.x):

```bash
pip install torch==2.2.1 torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/rocm6.0
```

### 4.2 Verify PyTorch GPU Access

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

Expected output:
```
PyTorch version: 2.2.1+rocm6.0
ROCm available: True
GPU name: AMD Radeon RX 7700 XT
VRAM: 12.0 GB
```

If `torch.cuda.is_available()` returns `False`:
1. Verify `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set
2. Verify user is in `render` group: `groups $USER` should include `render`
3. Restart terminal or system if needed

---

## Part 5: Clone & Setup Zubic et al. Repository

### 5.1 Clone Repository

```bash
cd ~
git clone https://github.com/uzh-rpg/ssms_event_cameras.git
cd ssms_event_cameras
```

### 5.2 Install Project Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pytorch-lightning` — training framework
- `hydra-core` — configuration management
- `einops` — tensor operations
- `torchmetrics` — evaluation metrics
- `h5py` & `hdf5plugin` — event data I/O
- `tqdm`, `matplotlib`, `numpy`, `pandas`

### 5.3 Install S5 State Space Model Package

```bash
pip install -e ./modules/s5
```

Verify:
```bash
python -c "from s5 import S5; print('S5 import OK')"
```

### 5.4 Install RVT Detection Library

```bash
pip install -e ./rpg_rvt/
```

---

## Part 6: Download & Prepare Gen1 Dataset

### 6.1 Register & Download Gen1

The Gen1 dataset requires registration with Prophesee.

1. **Register at:** https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/
2. **Download:** You'll receive a download link via email
3. **Download files:**
   - `test_a.7z` (21.5 GB)
   - `test_b.7z` (18.3 GB)
   - (Optional for training) `train_a.7z` through `train_e.7z` + `val_a.7z`, `val_b.7z`

```bash
# Create data directory
mkdir -p ~/data/gen1_raw
cd ~/data/gen1_raw

# Download using provided link (or use aria2c/wget if link provided)
# wget https://[prophesee-link]/test_a.7z
# wget https://[prophesee-link]/test_b.7z
```

### 6.2 Extract Archives

```bash
cd ~/data/gen1_raw

# Install 7z if not present
sudo apt install -y p7zip-full

# Extract test splits (needed for MVP evaluation)
7z x test_a.7z -o./
7z x test_b.7z -o./
```

Expected structure after extraction:
```
~/data/gen1_raw/
├── test/
│   ├── 17-04-06_15-09-57_3_td.dat.h5
│   ├── 17-04-06_15-09-57_3_td.dat_bbox.npy
│   └── ... (many more files)
```

### 6.3 Pre-process Dataset to HDF5 Voxel Format

```bash
cd ~/ssms_event_cameras

python RVT/preprocess/gen1.py \
  --data_dir ~/data/gen1_raw \
  --output_dir ~/data/gen1_processed \
  --num_workers 4
```

**Expected runtime:** 30–90 minutes depending on CPU speed.

Expected output:
```
Preprocessing Gen1 dataset...
Processing test split... [████████░░] 80%
Writing HDF5 voxels... Done
Output: ~/data/gen1_processed/
```

### 6.4 Verify Preprocessing

```bash
ls -lh ~/data/gen1_processed/
```

Should show `test/` subdirectory with HDF5 files.

---

## Part 7: Download Pre-trained S5-RVT Checkpoint

```bash
mkdir -p ~/checkpoints

# Download S5-ViT-Base checkpoint for Gen1 (~400 MB)
wget -O ~/checkpoints/s5_gen1_base.ckpt \
  https://download.ifi.uzh.ch/rpg/ssms_event_cameras/checkpoints/gen1/s5_vitb_gen1.ckpt

# Verify download (should be ~400 MB)
ls -lh ~/checkpoints/s5_gen1_base.ckpt
```

If URL is unavailable, check the repo README for updated links:
[https://github.com/uzh-rpg/ssms_event_cameras#pretrained-models](https://github.com/uzh-rpg/ssms_event_cameras#pretrained-models)

---

## Part 8: Run MVP Evaluation

### 8.1 Activate Environment

```bash
conda activate ssm_thesis
cd ~/ssms_event_cameras
```

### 8.2 Run Full Test-Set Evaluation

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python RVT/validation.py \
  dataset=gen1 \
  dataset.path=~/data/gen1_processed \
  checkpoint=~/checkpoints/s5_gen1_base.ckpt \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001
```

### 8.3 Expected Output

At the end of evaluation (20–40 minutes runtime):

```
┌─────────────────────────────────────────────────────────┐
│              Evaluation Results (Gen1 Test)              │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Class       │  AP@0.5      │  AP@0.5      │  Count     │
├──────────────┼──────────────┼──────────────┼────────────┤
│  Car         │  56.x        │  56.x        │  xxxxxx    │
│  Pedestrian  │  38.x        │  38.x        │  xxxxxx    │
├──────────────┼──────────────┼──────────────┼────────────┤
│  Overall     │  47.71       │              │            │
└──────────────┴──────────────┴──────────────┴────────────┘
```

### 8.4 Save Results

```bash
# Create results directory
mkdir -p ~/Thesis/results

# Save output to file
HSA_OVERRIDE_GFX_VERSION=11.0.0 python RVT/validation.py \
  dataset=gen1 \
  dataset.path=~/data/gen1_processed \
  checkpoint=~/checkpoints/s5_gen1_base.ckpt \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001 \
  2>&1 | tee ~/Thesis/results/gen1_s5_vitb_eval.txt
```

---

## Part 9: Troubleshooting

### GPU Not Detected by PyTorch

**Symptom:** `torch.cuda.is_available()` returns `False`

**Fixes:**
1. Verify environment variables:
   ```bash
   echo $HSA_OVERRIDE_GFX_VERSION  # Should print 11.0.0
   echo $ROCM_PATH                 # Should print /opt/rocm or similar
   ```

2. Verify group membership:
   ```bash
   groups $USER  # Should include 'render' and 'video'
   ```

3. Restart terminal or system after group changes:
   ```bash
   sudo usermod -aG render,video $USER
   # Log out and back in
   ```

### CUDA Out of Memory

**Symptom:** `RuntimeError: HIP out of memory`

**Fix:** Reduce evaluation batch size:
```bash
batch_size.eval=4  # or 2
```

### HDF5 Plugin Error During Preprocessing

**Symptom:** `ImportError: cannot import name 'bshuf'`

**Fix:**
```bash
pip uninstall hdf5plugin -y
pip install hdf5plugin==4.3.0
```

### Hydra Configuration Not Found

**Symptom:** `HydraException: Could not find config base.yaml`

**Fix:** Ensure you're running from the repo root:
```bash
cd ~/ssms_event_cameras
# Then run the evaluation command again
```

### mAP is 0.0 or NaN

**Causes:**
1. Dataset path is wrong — verify `dataset.path=~/data/gen1_processed` exists
2. Confidence threshold too high — use `model.postprocess.confidence_threshold=0.001`
3. Wrong checkpoint — verify `s5_gen1_base.ckpt` is for Gen1, not 1Mpx

---

## Part 10: Quick Reference Commands

### Daily Workflow

```bash
# Activate environment
conda activate ssm_thesis

# Verify GPU
rocm-smi

# Run evaluation
cd ~/ssms_event_cameras
HSA_OVERRIDE_GFX_VERSION=11.0.0 python RVT/validation.py \
  dataset=gen1 \
  dataset.path=~/data/gen1_processed \
  checkpoint=~/checkpoints/s5_gen1_base.ckpt \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001

# View results
cat ~/Thesis/results/gen1_s5_vitb_eval.txt
```

### If System Resets/Restarts

After a system restart:
1. Log in
2. `source ~/.bashrc` (initializes conda and HSA_OVERRIDE_GFX_VERSION)
3. `conda activate ssm_thesis`
4. Proceed with workflow above

---

## Part 11: Verification Checklist

Use this checklist before and after setup:

### Pre-Evaluation
- [ ] `rocm-smi` shows RX 7700 XT detected
- [ ] `$HSA_OVERRIDE_GFX_VERSION` is set to 11.0.0
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `~/data/gen1_processed/` exists with test/ subdirectory
- [ ] `~/checkpoints/s5_gen1_base.ckpt` exists (~400 MB)

### Post-Evaluation
- [ ] Evaluation script completed without errors
- [ ] Results table printed with mAP@0.5 value
- [ ] Overall mAP@0.5 is between 47.0 and 48.5
- [ ] Results saved to `~/Thesis/results/gen1_s5_vitb_eval.txt`

---

## Part 12: Integration with Thesis Report

Once evaluation is complete, use the recorded mAP value to populate the MVP section of main.tex. See `MVP_Report_Plan.md` in your Thesis folder for the LaTeX template and placeholder locations.

---

*Complete setup guide for Ubuntu 24.04 + AMD RX 7700 XT + Zubic et al. (2024) MVP reproduction.*
*Last updated: April 2026*
