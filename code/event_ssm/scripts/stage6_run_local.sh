#!/usr/bin/env bash
# Stage-6 SHORT training, local (RTX 5070 Ti). Registers the drop-in Mamba-2 backbone via the
# stage6_train.py launcher, then runs RVT train.py UNMODIFIED on the fixed-seed 10% Gen1 train subset.
# Progress bar is ON (Lightning default; train.py does not disable it) -- you will see a per-step
# tqdm bar with loss / it-s / ETA. bf16 autocast, no GradScaler (ISSUE-09). wandb offline.
set -euo pipefail

set +u                                               # conda's activate.d (cuda-nvcc) references unbound vars
source /home/ghost/miniforge3/etc/profile.d/conda.sh && conda activate events_signals
set -u
export WANDB_MODE=offline                            # RVT wandb cfg has no 'mode' key -> use the env var
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (real seq_len=21 is VRAM-heavy)
REPO=/home/ghost/Desktop/thesis-ssm-event-cameras
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"
cd "$REPO/external/ssms_event_cameras/RVT"          # hydra config_path="config" is relative to train.py

# ---------------- short-run knobs (edit freely) ----------------
DATASET="$REPO/data/gen1_subset10"   # build first: python -m event_ssm.integration.make_train_subset
MAX_STEPS=2000                       # short prelim; raise for a longer run
BATCH=4                              # bs8 OOMs at seq_len=21 on 16GB; 4 fits. Drop to 2 if still tight.
PRECISION="bf16-mixed"               # ISSUE-09; fp32 fallback: PRECISION=32
# ---------------------------------------------------------------

source "$REPO/code/event_ssm/scripts/stage6_overrides.sh"   # builds STAGE6_OVERRIDES from the knobs above
python "$REPO/code/event_ssm/scripts/stage6_train.py" "${STAGE6_OVERRIDES[@]}"
