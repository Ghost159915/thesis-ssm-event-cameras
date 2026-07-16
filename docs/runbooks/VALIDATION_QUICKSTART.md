# Validation Quickstart — Gen1 MVP (RTX 5070 Ti)

**For future re-runs only.** The environment is already built (`events_signals` conda env with the correct cu128/Blackwell stack). This is the short path to reproduce the Gen1 evaluation. For a from-scratch rebuild, see `MVP_Setup_Guide_Complete.md`.

## Run it (3 steps)

```bash
# 1. Activate the working environment
conda activate events_signals

# 2. Set paths (or add these to ~/.bashrc once)
export THESIS=~/Desktop/thesis-ssm-event-cameras
export REPO=$THESIS/external/ssms_event_cameras
export DATA_DIR=$THESIS/data/gen1_raw/gen1        # contains test/
export CKPT_PATH=$THESIS/checkpoints/gen1_base.ckpt
mkdir -p "$THESIS/results"

# 3. Run the Gen1 test-set evaluation
cd "$REPO"
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

**Expected:** runs ~15–35 min on the 5070 Ti, ends with a per-class AP table, **Overall mAP@0.5 ≈ 47.7** (success = 47.0–48.5). Results saved to `$THESIS/results/`.

## The two things that are easy to forget

1. **`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` must prefix the command.** Without it, PyTorch 2.6+ refuses to unpickle the Lightning checkpoint (`UnpicklingError: Weights only load failed`).
2. **The env already has `torchdata==0.9.0` pinned.** Do not `pip install -U torchdata` — newer versions delete the `datapipes` API the repo needs.

## If the env ever breaks or you move to Katana

Rebuild from the lock file (torch must come from the cu128 index first):

```bash
conda create -y -n events_signals python=3.11 && conda activate events_signals
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
pip install -r "$THESIS/requirements_5070ti_lock.txt"
pip install --no-deps "torchdata==0.9.0"
```

(On Katana, swap `hardware.gpus=0` for the assigned GPU id, and load the cluster's CUDA module instead of installing nvcc.)

## Sanity check (if results look wrong)

```bash
# GPU + Blackwell kernels present?
python -c "import torch; print(torch.__version__, torch.version.cuda, 'sm_120' in torch.cuda.get_arch_list())"
# Dataset present? (expect 470)
ls "$DATA_DIR/test" | wc -l
```

- mAP = 0.0 / NaN → `dataset.path` must point at the dir *containing* `test/`, not `test/` itself; keep `confidence_threshold=0.001`.
- `no kernel image is available` → wrong torch build; reinstall from the cu128 index.
