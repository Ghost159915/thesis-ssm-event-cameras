#!/usr/bin/env bash
# Stage-6 SHORT training, local (RTX 5070 Ti). Registers the drop-in Mamba-2 backbone via the
# stage6_train.py launcher, then runs RVT train.py UNMODIFIED on the fixed-seed 10% Gen1 train subset.
# Progress bar is ON (Lightning default; train.py does not disable it) -- you will see a per-step
# tqdm bar with loss / it-s / ETA. bf16 autocast, no GradScaler (ISSUE-09). wandb offline.
set -euo pipefail

source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
export WANDB_MODE=offline                            # RVT wandb cfg has no 'mode' key -> use the env var
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"          # hydra config_path="config" is relative to train.py

# ---------------- short-run knobs (edit freely) ----------------
DATASET="$REPO/data/gen1_subset10"   # build first: python -m event_ssm.integration.make_train_subset
MAX_STEPS=2000                       # short prelim; raise for a longer run
BATCH=8                              # lower to 4/2 if VRAM-bound (d_state=64)
PRECISION="bf16-mixed"               # ISSUE-09; fp32 fallback: PRECISION=32
# ---------------------------------------------------------------

python "$REPO/code/event_ssm/scripts/stage6_train.py" \
  dataset=gen1 model=rnndet +experiment/gen1=resnet_mamba \
  dataset.path="$DATASET" \
  training.precision="$PRECISION" \
  training.max_steps="$MAX_STEPS" training.max_epochs=1 \
  batch_size.train="$BATCH" batch_size.eval="$BATCH" \
  validation.val_check_interval="$MAX_STEPS" validation.check_val_every_n_epoch=null \
  hardware.gpus=0
