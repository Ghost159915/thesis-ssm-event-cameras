#!/usr/bin/env bash
# Stage-20 PureSSM 100k ANCHOR (RTX 5070 Ti by default). The first rung of the 100k ladder, PureSSM -> analog
# (cost of the LIF dynamics), needs PureSSM trained on the SAME 100k schedule as the spiking arms. The only PureSSM
# run so far is the Stage-14 400k one, and OneCycle stretches its learning-rate schedule over 400k steps, so it is
# not a 100k model (Stage-20 notes D1, D6). Sibling of stage20_full_local.sh, which is left untouched: it was
# mid-use by the running ladder chain when this anchor was added, and it only knows spiking arms.
#
# Recipe: PINNED and identical to the spiking arms (test-enforced: test_stage20_puressm_launcher.py runs both
# wrappers and compares the whole environment and the Hydra arguments they exec with) -- 100k steps, validation every 10k, batch 4, bf16-mixed, full val,
# full Gen1, +experiment/gen1=puressm (OmegaConf-equal to spikingssm except the model group, by test). Only the
# model differs. Compute-only knobs pass through exactly as in stage20_full_local.sh: NUM_WORKERS_TRAIN/
# NUM_WORKERS_EVAL (host RAM; they change the data order, so keep the defaults the ladder used), CHECKPOINT_BLOCKS
# (True|False; same maths, default True for the local 16 GB card), REPO, CONDA_SH, WANDB_MODE. Resume a crashed
# run with STAGE7_RESUME=/abs/last...ckpt; the resume guard refuses a checkpoint from another schedule length.
#
#   bash stage20_puressm_local.sh             # ~14 h locally
#   bash stage20_puressm_local.sh --cfg job   # GPU-free dry-run (compose + exit)
set -euo pipefail

CHECKPOINT_BLOCKS="${CHECKPOINT_BLOCKS:-True}"
case "$CHECKPOINT_BLOCKS" in
  True|False) ;;
  *) echo "[stage20-puressm] CHECKPOINT_BLOCKS must be True or False (got '${CHECKPOINT_BLOCKS}')" >&2; exit 2 ;;
esac

for a in "$@"; do                     # allow-list: dry-run and verbosity only (Hydra lets the LAST value win)
  case "$a" in
    --cfg|job|hydra.verbose=*) ;;
    *) echo "[stage20-puressm] argument '${a}' is not allowed: it could change the model, budget or labels of" \
            "a run labelled stage20_puressm_100k; use the env knobs" >&2
       exit 2 ;;
  esac
done

export REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
export EXPERIMENT=puressm
export MAX_STEPS=100000 VAL_EVERY=10000 BATCH=4
export PRECISION=bf16-mixed VAL_FRAC=1.0 MAX_EPOCHS=10000 DATASET="$REPO/data/gen1_raw/gen1"
export GROUP_NAME="stage20_puressm_100k"
export RUNDIR="$REPO/results/stage20/puressm_100k"
export PURESSM_MONITOR=1                        # every 200 backbone forwards, as in the spiking arms
export RESUME_EXPECT_TOTAL_STEPS="$MAX_STEPS"   # read by the resume guard in stage6_train.py (only on resume)
unset MAMBA_STEP_SCALE S5_STEP_SCALE            # contamination guard: Stage-9 inference-time dt hooks
unset SPIKING_ALLOW_ARM_OVERRIDE SPIKING_MONITOR   # spiking-only knobs; nothing to override or watch here

echo "[stage20-puressm] group=${GROUP_NAME} steps=${MAX_STEPS} val_every=${VAL_EVERY} batch=${BATCH}" \
     "checkpoint_blocks=${CHECKPOINT_BLOCKS}"
exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" \
  "model.backbone.checkpoint_blocks=${CHECKPOINT_BLOCKS}" \
  "$@"
