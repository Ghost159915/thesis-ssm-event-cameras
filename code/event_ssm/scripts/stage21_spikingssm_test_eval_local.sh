#!/usr/bin/env bash
# Stage-21 SpikingSSM TEST-set evaluation (RTX 5070 Ti). Sibling of stage14_puressm_test_eval_local.sh (unchanged):
# same recipe (scripts/stage7_eval.py -> RVT validation.py unmodified, use_test_set=1, bf16, confidence 0.001,
# per-class AP), with +experiment/gen1=spikingssm and the ablation ARM READ FROM THE CHECKPOINT
# (event_ssm/integration/spiking_ckpt.py): output mode, spiking stages, residual and readout settings come from the
# checkpoint's extra state, so a checkpoint cannot be evaluated as another arm by mistake (the Stage-18 D14 arm
# contract would refuse it anyway, but only after loading).
#
#   bash stage21_spikingssm_test_eval_local.sh /abs/path/epoch=...-val_AP=....ckpt
#   BATCH=2 bash stage21_spikingssm_test_eval_local.sh <ckpt>                      # if VRAM is tight
#   SPIKING_ALLOW_ARM_OVERRIDE=1 bash stage21_spikingssm_test_eval_local.sh <ckpt> model.backbone.spiking.threshold=0.8
#
# Extra arguments are allow-listed (spiking_ckpt.classify_extra_args): `--cfg job`, `hydra.verbose=...`, and a
# post-hoc sweep of a NON-learned threshold/beta with SPIKING_ALLOW_ARM_OVERRIDE=1. A learned knob is refused (the
# model would silently ignore the override), as is anything that changes the recipe. Post-hoc runs are filed under
# <arm tag>_posthoc/, never as the arm's canonical test eval.
# Log: results/stage21_test_eval/<arm tag>[_posthoc]/test_eval_<run id>_step<N>_<timestamp>.txt. Needs an idle GPU;
# the user runs it.
set -euo pipefail

set +u                                               # conda activate.d references unbound vars
source "${CONDA_SH:-/home/ghost/miniforge3/etc/profile.d/conda.sh}" && conda activate events_signals
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1            # PyTorch 2.6+ refuses to unpickle the Lightning ckpt otherwise
export PYTHONUNBUFFERED=1                            # stream the COCO eval / per-class AP table live
unset MAMBA_STEP_SCALE S5_STEP_SCALE                 # canonical 1x evaluation; Stage-9 hooks have their own launchers
REPO="${REPO:-/home/ghost/Desktop/thesis-ssm-event-cameras}"
export PYTHONPATH="$REPO/code:$REPO/external/ssms_event_cameras/RVT:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "usage: bash stage21_spikingssm_test_eval_local.sh <ckpt> [extra hydra overrides...]" >&2
  exit 2
fi
CKPT="$1"; shift
[[ "$CKPT" != /* ]] && CKPT="$PWD/$CKPT"
if [[ ! -f "$CKPT" ]]; then
  echo "[stage21-eval] ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi
DATASET="${DATASET:-$REPO/data/gen1_raw/gen1}"      # dir CONTAINING test/ (NOT test/ itself)
BATCH="${BATCH:-4}"

# arm + full evaluator argument list, both read from the checkpoint (fails here, before any evaluation, for a
# non-spiking or malformed checkpoint)
TAG="$(python -m event_ssm.integration.spiking_ckpt tag "$CKPT")"
VERDICT="$(python -m event_ssm.integration.spiking_ckpt check-extra "$CKPT" "$@")"   # exits 1 on a refused arg
[[ "$VERDICT" == "posthoc" ]] && TAG="${TAG}_posthoc"
EVAL_ARGV_TXT="$(python -m event_ssm.integration.spiking_ckpt argv "$CKPT" "$DATASET" "$BATCH")"
mapfile -t EVAL_ARGV <<< "$EVAL_ARGV_TXT"

CKPT_DIR="$(dirname "$CKPT")"
RUN_ID="ckpt"
[[ "$(basename "$CKPT_DIR")" == "checkpoints" ]] && RUN_ID="$(basename "$(dirname "$CKPT_DIR")")"
RESULTS="$REPO/results/stage21_test_eval/$TAG"
mkdir -p "$RESULTS"
STEP="NA"
[[ "$(basename "$CKPT")" =~ step=([0-9]+) ]] && STEP="${BASH_REMATCH[1]}"
OUT="$RESULTS/test_eval_${RUN_ID}_step${STEP}_$(date +%Y%m%d_%H%M%S).txt"

cd "$REPO/external/ssms_event_cameras/RVT"          # as the Stage-14 eval (run outputs land under RVT/)
{
  echo "[stage21-eval] ckpt:    $CKPT"
  echo "[stage21-eval] arm:     $TAG  (read from the checkpoint)"
  echo "[stage21-eval] dataset: $DATASET  (expects test/ with 470 recs)"
  echo "[stage21-eval] args:    ${EVAL_ARGV[*]} $*"
} | tee "$OUT"
python "$REPO/code/event_ssm/scripts/stage7_eval.py" "${EVAL_ARGV[@]}" "$@" 2>&1 | tee -a "$OUT"
