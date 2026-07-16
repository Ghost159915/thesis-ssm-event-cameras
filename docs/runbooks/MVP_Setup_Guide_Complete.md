# Complete MVP Setup Guide — From Zero to Evaluation
## Ubuntu 24.04 LTS (Noble) | NVIDIA RTX 5070 Ti (Blackwell)
### Thesis B — Benas Vaiciulis | UNSW MMAN4951

**Purpose:** Reproduce the Gen1 test-set evaluation of Zubic et al. (2024), *State Space Models for Event Cameras* (CVPR'24), on Ubuntu 24.04 with an **NVIDIA RTX 5070 Ti (16 GB, Blackwell, compute capability `sm_120`)**. This is a port of the original AMD RX 7700 XT / ROCm guide to the new CUDA system.

**The one thing that matters most:** Blackwell GPUs (`sm_120`) are **not** supported by the PyTorch build the upstream repo pins (`pytorch-cuda=11.8`, torch 2.2.1). They require a PyTorch wheel built against **CUDA 12.8+ (cu128)**, i.e. **torch ≥ 2.7**. Installing the upstream-pinned torch will produce `CUDA error: no kernel image is available for execution on the device`. Everything below installs the correct stack.

**Target result:** Overall **mAP@0.5 ≈ 47.7** on Gen1 (base model), matching the paper.

**Total time:** ~1–2 hours (the heavy dataset download/preprocessing was already done; `gen1.tar` is the pre-processed dataset).

---

## Part 0: Paths On This System

Everything lives under the thesis repo `~/Desktop/thesis-ssm-event-cameras` (the Zubic clone is under `external/`, large artifacts are gitignored). Define these once per shell session (or append to `~/.bashrc`):

```bash
export THESIS=~/Desktop/thesis-ssm-event-cameras
export REPO=$THESIS/external/ssms_event_cameras
export DATA_DIR=$THESIS/data/gen1_raw/gen1        # pre-processed Gen1 (contains test/)
export CKPT_PATH=$THESIS/checkpoints/gen1_base.ckpt
```

Confirm they exist:

```bash
ls -d "$REPO" "$DATA_DIR/test" "$CKPT_PATH"
```

All three should print without "No such file or directory". (The repo, the extracted `test/` split, and the 209 MB checkpoint are already on disk from the previous setup.)

---

## Part 1: System Dependencies

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y build-essential git curl wget gnupg2 cmake p7zip-full
```

`build-essential` provides `gcc`/`g++` (you have 13.3.0), which `pykeops` and `numba` use for JIT kernel compilation.

> **No ROCm, no kernel headers for ROCm, no `HSA_OVERRIDE_GFX_VERSION`.** All of that was AMD-specific and is removed. NVIDIA needs none of it.

---

## Part 2: NVIDIA Driver Verification (Already Installed)

The proprietary NVIDIA driver is already present and is new enough for Blackwell. Verify:

```bash
nvidia-smi
```

Expected (abbreviated):

```
NVIDIA-SMI 595.71.05   Driver Version: 595.71.05   CUDA Version: 13.2
NVIDIA GeForce RTX 5070 Ti      16303MiB
```

**Requirements for Blackwell:** driver **≥ 570** and a CUDA *runtime* of **≥ 12.8** (the "CUDA Version" shown by `nvidia-smi` is the max runtime the driver supports — 13.2 here, which is fine). If `nvidia-smi` fails or shows a driver < 570:

```bash
sudo ubuntu-drivers install        # installs the recommended driver
sudo reboot
```

> You do **not** need to install the standalone CUDA Toolkit (`nvcc`) system-wide. The PyTorch wheels bundle their own CUDA runtime. A toolkit is only needed for `pykeops`/`numba` JIT — installed into the conda env in Part 4.3, not system-wide.

---

## Part 3: Python Environment (Miniforge / Conda)

No conda is currently installed. Install Miniforge (conda-forge based, lightweight):

### 3.1 Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# Accept the location, type 'yes' to initialize conda
source ~/.bashrc
```

### 3.2 Create The Environment

Match the upstream Python version (3.11):

```bash
conda create -y -n events_signals python=3.11
conda activate events_signals
```

---

## Part 4: PyTorch (Blackwell / CUDA 12.8) — The Critical Step

### 4.1 Do NOT use the upstream conda install line

The repo's `installation_details.txt` and README say:

```bash
# ❌ DOES NOT WORK ON RTX 5070 Ti — pytorch-cuda=11.8 has no sm_120 kernels
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Skip it entirely.

### 4.2 Install the cu128 PyTorch wheels

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This pulls the latest stable PyTorch built against CUDA 12.8, which includes `sm_120` (Blackwell) kernels. (If you prefer to pin, any `torch>=2.7` from the `cu128` index works; newer is fine.)

### 4.3 Install a CUDA compiler into the env (for pykeops / numba)

`pykeops` JIT-compiles CUDA kernels at runtime and needs `nvcc`. Install a matching toolkit **inside the conda env** (keeps it isolated, no system pollution):

```bash
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
```

### 4.4 Verify PyTorch sees the GPU and has Blackwell kernels

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
cc = torch.cuda.get_device_capability(0)
print('Compute capability:', f'{cc[0]}.{cc[1]}')          # expect 12.0
print('Arch list:', torch.cuda.get_arch_list())            # must contain sm_120
# Real kernel test — this is what actually fails on a wrong build:
x = torch.randn(2048, 2048, device='cuda')
print('matmul OK:', bool((x @ x).sum().isfinite()))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"
```

Expected:

```
PyTorch: 2.x.x+cu128
CUDA available: True
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
Compute capability: 12.0
Arch list: [..., 'sm_90', 'sm_100', 'sm_120']
matmul OK: True
VRAM: 16.0 GB
```

If `get_arch_list()` does **not** contain `sm_120`, or the `matmul` line raises `no kernel image is available`, you installed a non-cu128 build — uninstall and redo 4.2:

```bash
pip uninstall -y torch torchvision torchaudio
```

---

## Part 5: Install Repository Dependencies

The repo has **no `requirements.txt`**; dependencies come from the README pip line. Install everything **except** torch (already installed in Part 4 — do not let these pull a CPU/old torch):

```bash
cd "$REPO"
pip install lightning wandb pandas plotly opencv-python tabulate pycocotools \
  bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py \
  hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn
```

### 5.1 Pin `torchdata` (REQUIRED — repo uses the removed DataPipes API)

The pip line above pulls the newest `torchdata` (≥0.10), which **deleted** the `torchdata.datapipes` module the repo imports (`ModuleNotFoundError: No module named 'torchdata.datapipes'`). Downgrade to **0.9.0** — the last release that still ships DataPipes — **without** disturbing torch:

```bash
pip install --no-deps "torchdata==0.9.0"
python -c "from torchdata.datapipes.map import MapDataPipe; print('datapipes OK')"
```

A deprecation `UserWarning` about datapipes is expected and harmless.

### 5.2 Re-confirm torch was not clobbered

```bash
python -c "import torch; assert 'cu128' in torch.__version__ or torch.version.cuda >= '12.8', torch.__version__; print('torch intact:', torch.__version__)"
```

If that assertion fails, reinstall the cu128 wheels (Part 4.2) — `lightning`/`torchdata` occasionally drag in a different torch.

> **W&B note:** evaluation does not require logging in to Weights & Biases, but the import must succeed. If prompted at runtime, run `wandb offline` once to disable uploads.

---

## Part 6: Dataset — Already Pre-processed

**No Prophesee registration, no preprocessing.** The `gen1.tar` you downloaded *is* the repo's pre-processed Gen1 dataset (the README "Required Data" link), and the **test split is already extracted** to:

```
$DATA_DIR/test/          # = ~/Documents/Thesis/data/gen1_raw/gen1/test/
```

Verify it has the expected sequence folders:

```bash
ls "$DATA_DIR/test" | head
ls "$DATA_DIR/test" | wc -l        # number of test sequences
```

You should see timestamped sequence directories (e.g. `17-04-13_15-05-43_4148500000_4208500000`). That is all the MVP test-set evaluation needs (`use_test_set=1`).

> Only `test/` is extracted (no `train/`/`val/`), which is correct for reproducing the reported test mAP. If you later need train/val, re-extract them from `gen1.tar`:
> ```bash
> tar -tf "$THESIS/data/gen1_raw/gen1.tar" | grep -E '/(train|val)/' | head   # inspect
> tar -xf "$THESIS/data/gen1_raw/gen1.tar" -C "$THESIS/data/gen1_raw"          # extract all
> ```

---

## Part 7: Pre-trained Checkpoint — Already Present

The Gen1 base checkpoint is already on disk:

```bash
ls -lh "$CKPT_PATH"        # ~209 MB, gen1_base.ckpt
```

If it were missing, re-download from the repo's Gen1 section:

```bash
mkdir -p "$THESIS/checkpoints"
wget -O "$CKPT_PATH" https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/gen1_base.ckpt
```

---

## Part 8: Run MVP Evaluation

### 8.1 Activate and position

```bash
conda activate events_signals
cd "$REPO"
```

### 8.2 Run the Gen1 test-set evaluation

This is the repo's canonical Gen1 command (README "Evaluation"), with `hardware.gpus=0` (single GPU, PCI bus id 0) and **no** `HSA_OVERRIDE_GFX_VERSION`:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python RVT/validation.py \
  dataset=gen1 \
  dataset.path="$DATA_DIR" \
  checkpoint="$CKPT_PATH" \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001
```

**Runtime:** ~15–35 minutes on the 5070 Ti. With 16 GB VRAM (vs the old 12 GB) `batch_size.eval=8` is comfortable; you can try `12` to go faster.

> **Why `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`?** PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`, which refuses to unpickle the config objects (`getattr`, OmegaConf, etc.) inside a full Lightning checkpoint — producing `UnpicklingError: Weights only load failed`. This env var restores `weights_only=False` for the loader. Safe here because the checkpoint is the official Zubic file. If a future Lightning passes `weights_only=True` explicitly (env var ignored), instead edit `RVT/validation.py` so the load call reads `Module.load_from_checkpoint(str(ckpt_path), weights_only=False, **{"full_config": config})`.

### 8.3 Expected output

A Lightning metrics table (verified output on the RTX 5070 Ti, June 2026):

```
       Test metric       DataLoader 0
   test/AP               0.4771549...   ← 47.7 mAP (COCO, IoU 0.50:0.95) — THE headline number
   test/AP_50            0.7527863...   ← 75.3 (AP@IoU 0.50 only)
   test/AP_75            0.4975681...   ← 49.8
   test/AP_L            0.5065961...
   test/AP_M            0.5492106...
   test/AP_S            0.3880771...
```

**Cite `test/AP` = 47.7 mAP** — this matches Zubic et al.'s reported value exactly. Do *not* quote `AP_50` (75.3) as the mAP; that is the IoU-0.5-only metric. Anything in **47.0–48.5** for `test/AP` is a successful reproduction. Runtime was ~14 min over 470 sequences at ~3.5 it/s.

### 8.4 Save results

```bash
mkdir -p "$THESIS/results"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python RVT/validation.py \
  dataset=gen1 \
  dataset.path="$DATA_DIR" \
  checkpoint="$CKPT_PATH" \
  use_test_set=1 \
  hardware.gpus=0 \
  +experiment/gen1="base.yaml" \
  batch_size.eval=8 \
  model.postprocess.confidence_threshold=0.001 \
  2>&1 | tee "$THESIS/results/gen1_s5_vitb_eval_5070ti.txt"
```

---

## Part 9: Troubleshooting (NVIDIA / Blackwell-specific)

### `CUDA error: no kernel image is available for execution on the device`
**Cause:** A non-cu128 torch (e.g. the upstream `pytorch-cuda=11.8` build, or a CPU wheel) — it has no `sm_120` kernels.
**Fix:** Reinstall the cu128 wheels (Part 4.2) and re-run the verification in 4.4. Confirm `sm_120` is in `torch.cuda.get_arch_list()`.

### `torch.cuda.is_available()` is `False`
1. `nvidia-smi` must work first (driver loaded). If it fails: `sudo ubuntu-drivers install` then reboot.
2. Make sure a CPU-only torch did not get installed by a dependency: `python -c "import torch; print(torch.version.cuda)"` should print `12.8`, not `None`.

### `pykeops` fails to compile / `Could not find nvcc`
**Cause:** No CUDA compiler in the env.
**Fix:** Install it into the env (Part 4.3): `conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8`. Then clear the pykeops cache and retry: `python -c "import pykeops; pykeops.clean_pykeops()"`.

### `numpy`-related errors (`np.float`/`np.int` removed, ABI warnings)
**Cause:** torch ≥2.7 ships with NumPy 2.x; some older code paths assume NumPy 1.x.
**Fix:** Pin NumPy down a major version in the env:
```bash
pip install "numpy<2"
```
(Re-verify torch still imports afterwards.)

### `lightning` import/API errors
**Cause:** The repo targets the 2.x `lightning` API from early 2024; a much newer release may have moved/renamed APIs.
**Fix:** Pin to a compatible 2.x:
```bash
pip install "lightning==2.2.5"
```

### `ModuleNotFoundError: No module named 'torchdata.datapipes'`
**Cause:** `torchdata ≥0.10` removed the DataPipes API the repo uses.
**Fix:** Pin to the last release that has it (Part 5.1): `pip install --no-deps "torchdata==0.9.0"`.

### `UnpicklingError: Weights only load failed` (at checkpoint load)
**Cause:** PyTorch 2.6+ defaults `torch.load` to `weights_only=True`; full Lightning checkpoints need full unpickling.
**Fix:** Prefix the eval command with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (already in Part 8). If ignored, edit the `load_from_checkpoint` call to pass `weights_only=False` (see Part 8.2 note).

### `[KeOps] CUDA libraries not found ... Switching to CPU only`
**Cause:** `pykeops` can't locate the CUDA toolkit at import.
**Status:** **Non-fatal** — the Gen1 S5-ViT eval does not depend on GPU KeOps, so the CPU fallback is fine and the run completes normally. Only if a code path actually needs GPU KeOps: `export CUDA_PATH=$CONDA_PREFIX` then `python -c "import pykeops; pykeops.clean_pykeops()"`.

### CUDA Out of Memory
16 GB is ample for Gen1 base, but if it OOMs (e.g. other GPU apps running):
```bash
batch_size.eval=4      # or 2
```
Also close other GPU consumers (browser, GNOME). Check with `nvidia-smi`.

### Hydra `Could not find config base.yaml`
Run from the repo root (`cd "$REPO"`) so Hydra resolves `+experiment/gen1="base.yaml"`.

### mAP is 0.0 or NaN
1. `dataset.path` must point at the directory **containing** `test/` — i.e. `$DATA_DIR`, not `$DATA_DIR/test`.
2. Keep `model.postprocess.confidence_threshold=0.001`.
3. Confirm the checkpoint is the **Gen1** base checkpoint (`gen1_base.ckpt`), not a 1 Mpx/gen4 one.

---

## Part 10: Quick Reference

```bash
# One-time per shell (or add to ~/.bashrc)
export THESIS=~/Documents/Thesis
export REPO=$THESIS/ssms_event_cameras
export DATA_DIR=$THESIS/data/gen1_raw/gen1
export CKPT_PATH=$THESIS/checkpoints/gen1_base.ckpt

# Daily workflow
conda activate events_signals
nvidia-smi                       # verify GPU
cd "$REPO"
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python RVT/validation.py \
  dataset=gen1 dataset.path="$DATA_DIR" checkpoint="$CKPT_PATH" \
  use_test_set=1 hardware.gpus=0 +experiment/gen1="base.yaml" \
  batch_size.eval=8 model.postprocess.confidence_threshold=0.001

cat "$THESIS/results/gen1_s5_vitb_eval_5070ti.txt"
```

After a reboot: log in → `conda activate events_signals` → re-export the paths (or rely on `~/.bashrc`) → run.

---

## Part 11: Verification Checklist

### Pre-Evaluation
- [ ] `nvidia-smi` shows RTX 5070 Ti, driver ≥ 570, CUDA ≥ 12.8
- [ ] `conda activate events_signals` works
- [ ] `torch.__version__` ends in `+cu128` and `get_arch_list()` contains `sm_120`
- [ ] GPU `matmul` test in Part 4.4 succeeds (no "no kernel image" error)
- [ ] `$DATA_DIR/test/` exists and lists sequence folders
- [ ] `$CKPT_PATH` exists (~209 MB)

### Post-Evaluation
- [ ] Script completed without errors
- [ ] Results table printed with mAP@0.5
- [ ] Overall mAP@0.5 in **47.0–48.5** (paper: 47.7)
- [ ] Output saved to `$THESIS/results/gen1_s5_vitb_eval_5070ti.txt`

---

## Part 12: Integration With Thesis Report

Once evaluation completes, use the recorded mAP to populate the MVP section of `main.tex`. See `MVP_Report_Plan.md` for the LaTeX template and placeholder locations.

---

## Appendix A: Known-Good Versions (RTX 5070 Ti, verified June 2026)

These are the exact versions that produced a successful Gen1 evaluation on this machine. Snapshot them with `pip freeze > requirements_5070ti_lock.txt`. When rebuilding from that lock file, install torch **first** from the cu128 index, because the `+cu128` tag only resolves there.

| Package | Version | Note |
|---|---|---|
| Python | 3.11.15 | conda env `events_signals` |
| torch / torchvision / torchaudio | 2.11.0+cu128 / 0.26.0+cu128 / 2.11.0+cu128 | `--index-url .../cu128` — Blackwell `sm_120` |
| cuda-nvcc (conda) | 12.8 | for pykeops/numba JIT |
| torchdata | **0.9.0** (`--no-deps`) | newer removes `datapipes` |
| lightning / pytorch-lightning | 2.6.5 | loaded checkpoint fine with the `weights_only` env var |
| numba / numpy | 0.65.1 / 2.4.4 | imported clean — no downgrade needed on this stack |
| hydra-core / omegaconf | 1.3.2 / 2.3.0 | |
| pandas | 3.0.3 | no eval-path break observed |
| pykeops | 2.3 | CPU fallback warning, non-fatal |

Rebuild in one shot later:
```bash
conda create -y -n events_signals python=3.11 && conda activate events_signals
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
pip install -r requirements_5070ti_lock.txt      # or the repo pip line + Part 5.1 torchdata pin
```

---

## Appendix B: What Changed From The AMD (ROCm) Guide

| Area | Old (RX 7700 XT / ROCm) | New (RTX 5070 Ti / CUDA) |
|---|---|---|
| GPU runtime | ROCm 6.4.2, `rocm-hip-sdk` | NVIDIA driver 595 (pre-installed) |
| Env override | `HSA_OVERRIDE_GFX_VERSION=11.0.0` | **None** (removed) |
| GPU groups | `usermod -aG render,video` | Not required |
| PyTorch | `torch==2.2.1 ... rocm6.0` | `torch>=2.7 ... cu128` (**Blackwell `sm_120`**) |
| Compiler for kernels | ROCm/HIP | `cuda-nvcc=12.8` in conda env (pykeops) |
| Dataset | Register + download + preprocess raw Gen1 | `gen1.tar` already pre-processed & extracted |
| GPU check | `rocm-smi` | `nvidia-smi` |
| OOM error | `RuntimeError: HIP out of memory` | `CUDA out of memory` |

---

*NVIDIA RTX 5070 Ti (Blackwell) + Ubuntu 24.04 + Zubic et al. (2024) Gen1 MVP reproduction.*
*Rewritten: June 2026. Supersedes the AMD RX 7700 XT / ROCm version.*
