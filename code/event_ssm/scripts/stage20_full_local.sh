#!/usr/bin/env bash
# Stage-20 FULL RUNS (RTX 5070 Ti by default). SpikingSSM at the ablation budget (100k) or the headline budget
# (400k) on full Gen1. Thin wrapper over stage7_midrun_local.sh, which keeps every OOM/resume/Hydra/live-bar fix;
# only the arm, the rung, the budget and the labels are set here. Sibling of stage19_short_local.sh (unchanged).
#
# Run plan (user decision 2026-10-07, docs/notes/Stage20_fullrun_notes.md): an equal-budget ladder at 100k
# (spike, graded, analog -- every arm on the SAME schedule length), then graded at 400k as the pre-registered
# headline. A 100k checkpoint taken from inside a 400k run is NOT a 100k arm: OneCycle stretches the learning-rate
# schedule over the run's total length, so at step 100k of 400k the rate is still near its peak.
#
# Recipe: PINNED, not env-overridable -- validation every 10k (as the PureSSM 400k run), batch 4, bf16-mixed,
# full val, full Gen1, +experiment/gen1=spikingssm (recipe-identical to PureSSM by test). A stale export must not
# leak into a run labelled as this one.
# Compute-only knobs MAY come from the environment, so the same script runs on a rented GPU: NUM_WORKERS_TRAIN/
# NUM_WORKERS_EVAL (host RAM; note they change the stream partition, hence the data order), CHECKPOINT_BLOCKS
# (True|False; activation recomputation, same maths, default True for the local 16 GB card), REPO, CONDA_SH,
# WANDB_MODE. Resume a crashed run with STAGE7_RESUME=/abs/last...ckpt, same ARM/STAGES/BUDGET. Guards: another
# arm is refused by the checkpoint's arm contract (Stage-18 D14); another rung fails the strict state-dict load;
# another BUDGET is refused by the resume guard (event_ssm/integration/resume_guard.py, via the exported
# RESUME_EXPECT_TOTAL_STEPS), because a full-state resume would otherwise restore the other run's OneCycle schedule.
#
# Labelling: the W&B group and run dir are DERIVED, stage20_<arm>_s<stages>_<budget>; extra arguments are
# allow-listed (`--cfg job`, `hydra.verbose=...`) because Hydra lets the LAST value of a key win.
#
#   ARM=spike  STAGES=2,3,4 BUDGET=100k bash stage20_full_local.sh           # ~14 h locally
#   ARM=graded STAGES=2,3,4 BUDGET=400k bash stage20_full_local.sh           # ~56 h locally
#   ARM=spike  STAGES=2,3,4 BUDGET=100k bash stage20_full_local.sh --cfg job # GPU-free dry-run (compose + exit)
set -euo pipefail

ARM="${ARM:-}"
STAGES="${STAGES:-}"
BUDGET="${BUDGET:-}"
CHECKPOINT_BLOCKS="${CHECKPOINT_BLOCKS:-True}"
case "$ARM" in
  analog|graded|spike) ;;
  *) echo "[stage20] ARM must be analog, graded or spike (got '${ARM}')" >&2; exit 2 ;;
esac
case "$STAGES" in                     # ladder rungs only (spiking_stages ⊆ temporal stages 2-4)
  4|3,4|2,3,4) ;;
  *) echo "[stage20] STAGES must be one ladder rung: 4 | 3,4 | 2,3,4 (got '${STAGES}')" >&2; exit 2 ;;
esac
case "$BUDGET" in
  100k) STEPS=100000 ;;
  400k) STEPS=400000 ;;
  *) echo "[stage20] BUDGET must be 100k (ablation ladder) or 400k (headline) (got '${BUDGET}')" >&2; exit 2 ;;
esac
case "$CHECKPOINT_BLOCKS" in
  True|False) ;;
  *) echo "[stage20] CHECKPOINT_BLOCKS must be True or False (got '${CHECKPOINT_BLOCKS}')" >&2; exit 2 ;;
esac

for a in "$@"; do                     # allow-list: dry-run and verbosity only (see header)
  case "$a" in
    --cfg|job|hydra.verbose=*) ;;
    *) echo "[stage20] argument '${a}' is not allowed: it could change the arm, budget or labels of a run" \
            "labelled stage20_<arm>_s<stages>_<budget>; use the env knobs or make it its own labelled arm" >&2
       exit 2 ;;
  esac
done

TAG="${ARM}_s${STAGES//,/}_${BUDGET}"
export REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
export EXPERIMENT=spikingssm
export MAX_STEPS="$STEPS" VAL_EVERY=10000 BATCH=4
export PRECISION=bf16-mixed VAL_FRAC=1.0 MAX_EPOCHS=10000 DATASET="$REPO/data/gen1_raw/gen1"
export GROUP_NAME="stage20_${TAG}"
export RUNDIR="$REPO/results/stage20/${TAG}"
export SPIKING_MONITOR=1 PURESSM_MONITOR=1      # every 200 backbone forwards (validation included)
export RESUME_EXPECT_TOTAL_STEPS="$MAX_STEPS"   # read by the resume guard in stage6_train.py (only on resume)
unset MAMBA_STEP_SCALE S5_STEP_SCALE            # contamination guard: Stage-9 inference-time dt hooks
unset SPIKING_ALLOW_ARM_OVERRIDE                # Stage-22 eval-only override; never wanted in a training launch

echo "[stage20] arm=${TAG} group=${GROUP_NAME} steps=${MAX_STEPS} val_every=${VAL_EVERY} batch=${BATCH}" \
     "checkpoint_blocks=${CHECKPOINT_BLOCKS}"
exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" \
  "model.backbone.spiking.output_mode=${ARM}" \
  "model.backbone.spiking.spiking_stages=[${STAGES}]" \
  "model.backbone.checkpoint_blocks=${CHECKPOINT_BLOCKS}" \
  "$@"
