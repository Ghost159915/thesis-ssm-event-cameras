#!/usr/bin/env bash
# Stage-19 LOCAL SHORT RUN (RTX 5070 Ti). SpikingSSM, 25k steps on full Gen1: the kill-switch evidence
# ("is the spiking model training stably?"). Thin wrapper over stage7_midrun_local.sh, which keeps every
# OOM/resume/Hydra/live-bar fix; only the arm, the budget and the labels are set here.
#
# Comparability: MAX_STEPS/VAL_EVERY/BATCH are PINNED to the Stage-13 PureSSM short run (25k / 5k / 4,
# OneCycle compressed into 25k steps; val/AP 0.351) -> a same-recipe anchor, and so are the launcher's
# other recipe knobs (PRECISION, VAL_FRAC, MAX_EPOCHS, DATASET, EXPERIMENT). None is env-overridable: a
# stale `export MAX_STEPS=400000` or `PRECISION=32` must not leak into a run labelled as this one.
# checkpoint_blocks=True is the local-16GB fallback (activation recomputation: same maths, less VRAM).
# Workers stay overridable (NUM_WORKERS_TRAIN/EVAL, host RAM) but are NOT result-neutral: under mixed
# sampling they set how stream sequences are partitioned across workers, so the data ORDER differs from
# the Stage-13 anchor (6/2 workers there, 2/1 here). The batch mixture (2 stream + 2 random) does not.
#
# Labelling (Stage-18 notes §8.1): the arms differ only by CLI overrides, so the W&B group and the run
# dir are DERIVED from the arm here and cannot be set by hand: stage19_short_<arm>_s<stages>. For the
# same reason extra arguments are allow-listed (`--cfg job`, `hydra.verbose=...`): Hydra lets the LAST
# value of a key win, so e.g. a trailing `model.backbone.spiking.residual=True` would otherwise run a
# different arm under this arm's label. Checkpoints additionally carry their arm (Stage-18 D14).
#
#   ARM=spike  STAGES=4 bash stage19_short_local.sh              # hours: see the Stage-19 notes
#   ARM=graded STAGES=4 bash stage19_short_local.sh
#   ARM=spike  STAGES=4 bash stage19_short_local.sh --cfg job    # GPU-free dry-run (compose + exit)
#   ARM=spike  STAGES=4 STAGE7_RESUME=/abs/last...ckpt bash stage19_short_local.sh   # resume a crash
#
# Kill-switch criterion (pre-registered 2026-10-06, before any run): finite loss throughout; for the
# spike/graded arms no SILENT/SATURATED [spk-monitor] line after warm-up (analog emits the membrane,
# not the spike, so its rate is reported only); val/AP rising across the 5 checkpoints; final
# val/AP >= 0.15 (the Stage-13 gate). The gap to PureSSM's 0.351 is reported, not gated.
set -euo pipefail

ARM="${ARM:-}"
STAGES="${STAGES:-4}"                 # the ladder start
case "$ARM" in
  analog|graded|spike) ;;
  *) echo "[stage19] ARM must be analog, graded or spike (got '${ARM}')" >&2; exit 2 ;;
esac
case "$STAGES" in                     # ladder rungs only (spiking_stages ⊆ temporal stages 2-4)
  4|3,4|2,3,4) ;;
  *) echo "[stage19] STAGES must be one ladder rung: 4 | 3,4 | 2,3,4 (got '${STAGES}')" >&2; exit 2 ;;
esac

for a in "$@"; do                     # allow-list: dry-run and verbosity only (see header)
  case "$a" in
    --cfg|job|hydra.verbose=*) ;;
    *) echo "[stage19] argument '${a}' is not allowed: it could change the arm, budget or labels" \
            "of a run labelled stage19_short_<arm>_s<stages>; make it its own labelled arm instead" >&2
       exit 2 ;;
  esac
done

TAG="${ARM}_s${STAGES//,/}"
export REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
export EXPERIMENT=spikingssm
export MAX_STEPS=25000 VAL_EVERY=5000 BATCH=4
export PRECISION=bf16-mixed VAL_FRAC=1.0 MAX_EPOCHS=10000 DATASET="$REPO/data/gen1_raw/gen1"
export GROUP_NAME="stage19_short_${TAG}"
export RUNDIR="$REPO/results/stage19/${TAG}"
export SPIKING_MONITOR=1 PURESSM_MONITOR=1      # every 200 backbone forwards (validation included)
unset MAMBA_STEP_SCALE S5_STEP_SCALE            # contamination guard: Stage-9 inference-time dt hooks

echo "[stage19] arm=${TAG} group=${GROUP_NAME} steps=${MAX_STEPS} val_every=${VAL_EVERY} batch=${BATCH}"
exec bash "$REPO/code/event_ssm/scripts/stage7_midrun_local.sh" \
  "model.backbone.spiking.output_mode=${ARM}" \
  "model.backbone.spiking.spiking_stages=[${STAGES}]" \
  model.backbone.checkpoint_blocks=True \
  "$@"
